#!/usr/bin/env python3
"""Vast.ai control from the Mac: the same worker image, R2 storage and job lists as salad.py, on rented Vast machines.

Vast lists every machine with its own price, internet speed and internet price (some hosts charge $0.05/GB, which
on ~80 GB of models costs more than the GPU), so each run rents the machine with the lowest ESTIMATED TOTAL cost:
GPU hours at that card's speed + the download time + the download and upload charges. Never above --max-price per
hour (default $0.50; RunPod was $1.70) or --max-gb ($ per GB of download, default $0.01). The worker destroys its
own instance when it finishes (nothing keeps billing); `watch` also destroys it when the run is done, replaces a
machine that is too slow or stuck, and enforces the time limit.

  .venv/bin/python scripts/salad/vast.py offers gen|train [--kinds t2i,i2v]   the cheapest machines and their estimated cost
  .venv/bin/python scripts/salad/vast.py gen <run> <jobs.json> [--machines 3] [--bid 1.2 --gpus "RTX 5090"]
        upload jobs, rent the cheapest machine(s); --bid = interruptible (about half price; a bumped machine is replaced
        by `watch` and the next one skips every output already in R2)
  .venv/bin/python scripts/salad/vast.py train mae,ray                         one machine per LoRA
  .venv/bin/python scripts/salad/vast.py watch <run> [--minutes 90]            babysit a run until it's done (run in background)
  .venv/bin/python scripts/salad/vast.py status | stop <run> | credit

Status, outputs and LoRAs are read with salad.py (status / pull / pull-lora): they live in R2 either way.
"""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import salad as S  # noqa: E402  (R2 helpers, job upload, image name)

API = "https://console.vast.ai/api/v0"
LOGS = os.path.join(S.ROOT, "avatars", "lora", "salad-logs")
BAD = os.path.join(LOGS, "vast-bad-machines.json")
# time a job takes relative to an RTX 4090 (measured on Salad: 5090 ~0.8 s/step, 3090 ~2-3 s/step on LoRA training)
SPEED = {"RTX 5090": 0.7, "RTX 4090": 1.0, "RTX 4090D": 1.1, "L40S": 1.0, "RTX 6000Ada": 1.0, "L40": 1.1, "RTX PRO 6000": 0.7,
         "RTX 5880Ada": 1.1, "RTX 3090 Ti": 1.6, "RTX 3090": 1.8, "RTX A6000": 1.7, "A40": 1.8, "RTX PRO 4000": 1.8,
         "RTX A5000": 2.0, "A10": 2.6, "RTX 4000Ada": 2.4}
# seconds per job on a 4090, and the image + models each run downloads
JOB_S = {"edit": 8, "t2i": 12, "i2v": 150, "talk": 240}
IMAGE_GB = 10
TRAIN_H = 0.45  # one LoRA, 1200 steps, on a 4090


def key():
    k = S.E.get("VAST_API_KEY")
    if not k:
        sys.exit("Put VAST_API_KEY in .env")
    return k


def vast(method, path, body=None):
    # paths under /v1/ (the instance list moved there) are given from the api root
    url = API.removesuffix("/v0") + path if path.startswith("/v1/") else API + path
    req = urllib.request.Request(url, method=method, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Authorization": f"Bearer {key()}", "Content-Type": "application/json", "User-Agent": "mae-vast/1.0"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                t = r.read()
                return json.loads(t) if t else {}
        except urllib.error.HTTPError as e:
            if e.code < 500 or attempt == 3:
                raise SystemExit(f"Vast API {method} {path}: HTTP {e.code} {e.read().decode()[:400]}")
        except urllib.error.URLError as e:
            if attempt == 3:
                raise SystemExit(f"Vast API {method} {path}: {e}")
        time.sleep(10)


def bad_machines():
    return set(json.load(open(BAD))) if os.path.exists(BAD) else set()


def mark_bad(machine_id, why):
    bad = bad_machines() | {machine_id}
    os.makedirs(LOGS, exist_ok=True)
    json.dump(sorted(bad), open(BAD, "w"))
    print(f"machine {machine_id} marked bad ({why}); it won't be rented again", flush=True)


def plan(kinds, train, njobs):
    """(disk GB, download GB, upload GB, GPU hours on a 4090) for a run."""
    if train:
        models = S.MODEL_GB["t2i"] + 12  # Z-Image Turbo + the training adapter and text encoder in full precision
        return 120, IMAGE_GB + models, 3.0, TRAIN_H
    models = sum(S.MODEL_GB[k] for k in kinds)
    hours = 0.15 + sum(JOB_S[k] * n for k, n in njobs.items()) / 3600
    return 25 + models + IMAGE_GB, IMAGE_GB + models, 0.2 + 0.05 * sum(njobs.values()), hours


def speed(gpu):
    return SPEED.get(gpu.replace("_", " "), 2.0)


def offers(disk, down_gb, up_gb, hours, max_price, max_gb, min_vram, bid=0.0, gpus=None):
    """bid > 0: interruptible machines too, bidding that factor over the host's minimum (1.2 = 20% above, fewer bumps);
    both kinds are compared and the cheapest total wins (a machine is sometimes cheaper on-demand than by bidding)."""
    if bid:
        od = offers(disk, down_gb, up_gb, hours, max_price, max_gb, min_vram, 0.0, gpus)
        # interruptible only when clearly cheaper (25%+ under the best on-demand total): on 2026-10-18 two of three
        # bids never started and several cost the same as on-demand, so a small saving isn't worth the risk
        cap = 0.75 * od[0]["est_cost"] if od else float("inf")
        both = od + [o for o in _offers(disk, down_gb, up_gb, hours, max_price, max_gb, min_vram, bid, gpus) if o["est_cost"] <= cap]
        # on-demand first on a tie: it can't be bumped
        return sorted(both, key=lambda o: (round(o["est_cost"], 2), o.get("is_bid_offer", False)))
    return _offers(disk, down_gb, up_gb, hours, max_price, max_gb, min_vram, 0.0, gpus)


def _offers(disk, down_gb, up_gb, hours, max_price, max_gb, min_vram, bid, gpus):
    q = {"num_gpus": {"eq": 1}, "gpu_ram": {"gte": min_vram * 1000}, "compute_cap": {"gte": 800}, "cuda_max_good": {"gte": 13.0},
         "rentable": {"eq": True}, "rented": {"eq": False}, "disk_space": {"gte": disk}, "reliability2": {"gte": 0.97},
         "inet_down": {"gte": 400}, "inet_down_cost": {"lte": max_gb},
         "type": "bid" if bid else "ondemand", "allocated_storage": disk, "order": [["dph_total", "asc"]], "limit": 400}
    if gpus:
        q["gpu_name"] = {"in": [g.strip() for g in gpus.split(",")]}
    out = []
    skip = bad_machines()
    for o in vast("POST", "/bundles/", q).get("offers", []):
        # hosts in mainland China often can't reach HuggingFace or GitHub's image registry
        if o["machine_id"] in skip or str(o.get("geolocation", "")).endswith(", CN"):
            continue
        # a bid pays our price for the GPU plus the host's storage share (dph_total - dph_base), never the on-demand rate
        price = round(o["min_bid"] * bid + (o["dph_total"] - o["dph_base"]), 4) if bid else o["dph_total"]
        if price > max_price:
            continue
        o = dict(o, price=price, is_bid_offer=bool(bid))
        dl_h = down_gb * 8 / (o["inet_down"] * 0.5) / 3600  # at half the advertised speed
        gpu_h = hours * speed(o["gpu_name"])
        cost = price * (gpu_h + dl_h) + down_gb * o["inet_down_cost"] + up_gb * o["inet_up_cost"]
        out.append(dict(o, est_cost=cost, est_hours=gpu_h + dl_h))
    return sorted(out, key=lambda o: o["est_cost"])


def show(o):
    return (f"{o['gpu_name']} in {o.get('geolocation') or '?'} · ${o['price']:.3f}/hr{' interruptible' if o.get('is_bid_offer') else ''} · internet {int(o['inet_down'])} Mbps, "
            f"${o['inet_down_cost'] * 1000:.2f}/TB down · est. {o['est_hours'] * 60:.0f} min, ${o['est_cost']:.2f} total")


def rent(run, mode, extra_env, disk, down_gb, up_gb, hours, a):
    label = S.group_name(run)
    if instances(label):
        sys.exit(f"{label} already has a machine (vast.py stop {run} first)")
    for o in offers(disk, down_gb, up_gb, hours, a.max_price, a.max_gb, a.min_vram, a.bid, a.gpus)[:5]:
        envs = {k: S.E[k] for k in ("R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET")}
        envs.update(MODE=mode, RUN=run, MAX_MINUTES=str(a.minutes), HF_TOKEN=S.E.get("HF_TOKEN", ""), VAST_MACHINE_ID=str(o["machine_id"]), **extra_env)
        body = {"client_id": "me", "image": S.IMAGE, "env": envs, "disk": disk, "label": label, "runtype": "args",
                "args": ["/opt/comfy-venv/bin/python", "/opt/worker/main.py"]}
        if o.get("is_bid_offer"):
            body["price"] = round(o["min_bid"] * a.bid, 4)
            o["bid_price"] = body["price"]
        try:
            r = vast("PUT", f"/asks/{o['id']}/", body)
        except SystemExit as e:
            print(f"offer {o['id']} gone ({str(e)[:120]}); next", flush=True)
            continue
        if r.get("success"):
            print(f"Rented {label}: {show(o)} (instance {r['new_contract']}, machine {o['machine_id']})", flush=True)
            return o
    sys.exit("No machine under the price limits right now; nothing rented.")


def instances(label=None):
    items = vast("GET", "/v1/instances/").get("instances", [])
    return [i for i in items if label is None or i.get("label") == label]


def destroy(label):
    for i in instances(label):
        vast("DELETE", f"/instances/{i['id']}/")
        print(f"destroyed {label} (instance {i['id']})", flush=True)


def credit():
    return vast("GET", "/users/current/").get("credit")


def gen_plan(jobs_path):
    spec = json.load(open(jobs_path))
    njobs = {}
    for j in spec["jobs"]:
        njobs[j["kind"]] = njobs.get(j["kind"], 0) + 1
    return set(njobs), njobs


def cmd_offers(a):
    if a.what == "train":
        p = plan(set(), True, {})
    else:
        kinds = set(a.kinds.split(","))
        p = plan(kinds, False, {k: 10 for k in kinds})
    disk, down, up, hours = p
    print(f"disk {disk} GB, download ~{down} GB, ~{hours * 60:.0f} min of 4090 time")
    for o in offers(disk, down, up, hours, a.max_price, a.max_gb, a.min_vram, a.bid, a.gpus)[:12]:
        print("  " + show(o))


def jobs_copy(run):
    return os.path.join(LOGS, f"vast-{run}.jobs.json")


def split(jobs, n):
    """n balanced job lists (by expected time); a clip stays with the keyframe it is animated from."""
    groups = {}
    for j in jobs:
        groups.setdefault(j["id"].removesuffix("-key"), []).append(j)
    parts = [[] for _ in range(n)]
    load = [0.0] * n
    for g in sorted(groups.values(), key=lambda g: -sum(JOB_S[j["kind"]] for j in g)):
        k = load.index(min(load))
        parts[k] += g
        load[k] += sum(JOB_S[j["kind"]] for j in g)
    return [p for p in parts if p]


def by_kind(jobs):
    """One list per kind of work, so each machine downloads only its own models: talking shots (LTX-2.3), clips with
    their keyframes (Z-Image + LTX-2.5), stills (Z-Image only, ~30 GB: ready in minutes)."""
    clips = {j["id"] for j in jobs if j["kind"] == "i2v"}
    parts = {"talk": [], "clips": [], "stills": []}
    for j in jobs:
        if j["kind"] == "talk":
            parts["talk"].append(j)
        elif j["kind"] == "i2v" or j["id"].removesuffix("-key") in clips and j["id"].endswith("-key"):
            parts["clips"].append(j)
        else:
            parts["stills"].append(j)
    return {k: v for k, v in parts.items() if v}


def cmd_gen(a):
    """--split kind: one machine per kind of work (runs <run>-talk, <run>-clips, <run>-stills);
    --machines N: N balanced mixed runs <run>-1 … <run>-N. Outputs: salad.py pull each run into the same folder."""
    spec = json.load(open(a.jobs))
    if a.split == "kind":
        parts = by_kind(spec["jobs"])
    elif a.machines > 1:
        parts = {str(k): p for k, p in enumerate(split(spec["jobs"], a.machines), 1)}
    else:
        parts = {"": spec["jobs"]}
    os.makedirs(LOGS, exist_ok=True)
    for name, part in parts.items():
        run = f"{a.run}-{name}" if name else a.run
        path = jobs_copy(run)
        json.dump(dict(spec, jobs=part), open(path, "w"), indent=1)
        kinds, njobs = gen_plan(path)
        S.upload_jobs(run, path)
        disk, down, up, hours = plan(kinds, False, njobs)
        rent(run, "gen", {}, disk, down, up, hours, a)
        print(f"  {run}: {len(part)} jobs {njobs}", flush=True)


def cmd_train(a):
    for n in a.names.split(","):
        if not S.listing(f"lora/{n}/data/"):
            sys.exit(f"{n}: no training set in R2 yet (salad.py push-lora {n})")
    disk, down, up, hours = plan(set(), True, {})
    for n in a.names.split(","):
        rent(f"train-{n}", "train", {"LORAS": n}, disk, down, up, hours, a)


def read_status(run):
    try:
        return json.loads(S.s3.get_object(Bucket=S.E["R2_BUCKET"], Key=f"status/{run}.json")["Body"].read())
    except Exception:
        return {}


def cmd_watch(a):
    """Every minute: done → destroy and stop; slow/broken machine → mark it bad, rent the next cheapest; stuck loading
    (image or models downloading too slowly) for 25 min → same; past --minutes → destroy and stop."""
    label = S.group_name(a.run)
    mode = "train" if a.run.startswith("train-") else "gen"
    t0, c0 = time.time(), credit()
    print(f"watching {label}; credit ${c0:.2f}", flush=True)
    rents, last_seen, down_since, crashed_on = 0, {}, None, ""
    jobs = a.jobs or jobs_copy(a.run)
    while True:
        st = read_status(a.run)
        inst = instances(label)
        i = inst[0] if inst else None
        mins = (time.time() - t0) / 60
        state = st.get("state", "no status")
        print(f"[{time.strftime('%H:%M')}] {state} · {(i or {}).get('actual_status', 'no machine')} · "
              f"{st.get('done', '')}/{st.get('total', '')} {st.get('progress', '')[:60]}", flush=True)
        for line in st.get("log", [])[-3:]:
            if line not in last_seen:
                last_seen[line] = 1
                print("   " + line, flush=True)
        # only a worker started after this watch counts (an old machine's heartbeat keeps "updated" fresh)
        fresh = st.get("started", "") >= time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0 - 60))
        if state == "crashed" and fresh and rents < 4 and str(st.get("machine", "")) != crashed_on:
            # a crash before the work is done (a host's disk I/O error, a broken download) is the machine's fault
            # more often than ours: try another machine; the same crash on a second machine stops the run
            crashed_on = str(st.get("machine", ""))
            print("crashed: " + (st.get("log") or ["?"])[-1].strip().splitlines()[-1][:160], flush=True)
            if crashed_on.isdigit():
                mark_bad(int(crashed_on), "crashed")
            destroy(label)
            kinds, njobs = gen_plan(jobs) if mode == "gen" else (set(), {})
            disk, down, up, hours = plan(kinds, False, njobs) if mode == "gen" else plan(set(), True, {})
            rent(a.run, mode, {"LORAS": a.run[len("train-"):]} if mode == "train" else {}, disk, down, up, hours, a)
            rents += 1
            time.sleep(60)
            continue
        if state in ("done", "done with failures", "failed", "crashed") and fresh:
            destroy(label)
            print(f"FINISHED ({state}); credit ${credit():.2f} (was ${c0:.2f})", flush=True)
            return
        if mins > a.minutes:
            destroy(label)
            print(f"TIME LIMIT; credit ${credit():.2f} (was ${c0:.2f})", flush=True)
            return
        replace = None
        if state == "slow machine" and st.get("bad_machine"):
            replace = (st["bad_machine"], "too slow")
        elif i and any(x in str(i.get("status_msg") or "") for x in ("Error response from daemon", "nvidia-container-cli", "OCI runtime")):
            # the host can't start containers on its GPU (seen 2026-10-18): broken, never rent it again
            replace = (str(i.get("machine_id")), "host error: " + str(i.get("status_msg"))[:80].replace("\n", " "))
        elif i and i.get("intended_status") == "stopped":
            # Vast stopped it (an interruptible machine that was outbid): rent another, on-demand
            replace = ("", "stopped by Vast (outbid)")
            a.bid = 0.0
        elif i and i.get("actual_status") in ("created", None) and time.time() - i.get("start_date", time.time()) > 10 * 60:
            # never started: usually an interruptible bid that doesn't win the machine; go on-demand from now on
            replace = ("", "never started (bid not winning?)")
            a.bid = 0.0
        elif i and i.get("actual_status") == "loading" and time.time() - i.get("start_date", time.time()) > 25 * 60:
            replace = (str(i.get("machine_id")), "still downloading after 25 min")
        elif i and i.get("actual_status") not in ("running", "loading", "created", None):
            # interruptible: an outbid machine stops ("exited"/"stopped"); an offline host is broken
            down_since = down_since or time.time()
            if i.get("actual_status") == "offline" or time.time() - down_since > 3 * 60:
                why = f"machine {i.get('actual_status')} ({str(i.get('status_msg', ''))[:60]})"
                replace = (str(i.get("machine_id")) if i.get("actual_status") == "offline" else "", why)
        elif not i and rents and mins > 3:
            replace = ("", "machine gone")
        if i and i.get("actual_status") in ("running", "loading", "created"):
            down_since = None
        if replace:
            if rents >= 4:
                destroy(label)
                print("GAVE UP after 4 machines", flush=True)
                return
            print(f"replacing the machine: {replace[1]}", flush=True)
            if replace[0]:
                mark_bad(int(replace[0]) if replace[0].isdigit() else replace[0], replace[1])
            destroy(label)
            if mode == "gen":
                kinds, njobs = gen_plan(jobs)
                disk, down, up, hours = plan(kinds, False, njobs)
            else:
                disk, down, up, hours = plan(set(), True, {})
            extra = {"LORAS": a.run[len("train-"):]} if mode == "train" else {}
            rent(a.run, mode, extra, disk, down, up, hours, a)
            rents += 1
            down_since = None
        elif i:
            rents = max(rents, 1)
        time.sleep(60)


def cmd_status(a):
    for i in instances():
        print(f"{i.get('label')}: {i.get('actual_status')} · {i.get('gpu_name')} · ${i.get('dph_total', 0):.3f}/hr · {(i.get('status_msg') or '')[:80]}")
    print(f"credit ${credit():.2f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    o = sp.add_parser("offers"); o.add_argument("what", choices=["gen", "train"]); o.add_argument("--kinds", default="t2i,i2v")
    g = sp.add_parser("gen"); g.add_argument("run"); g.add_argument("jobs"); g.add_argument("--machines", type=int, default=1); g.add_argument("--split", choices=["kind"])
    t = sp.add_parser("train"); t.add_argument("names")
    w = sp.add_parser("watch"); w.add_argument("run"); w.add_argument("--jobs")
    for q in (o, g, t, w):
        q.add_argument("--max-price", type=float, default=0.50, help="$/hr cap (RunPod was $1.70)")
        q.add_argument("--max-gb", type=float, default=0.01, help="$ per GB of download cap ($10/TB)")
        q.add_argument("--min-vram", type=int, default=24, help="GB")
        q.add_argument("--bid", type=float, default=0.0, help="interruptible: bid this factor over the minimum (e.g. 1.2); 0 = on-demand")
        q.add_argument("--gpus", help='only these cards, e.g. "RTX 5090,RTX 4090"')
        q.add_argument("--minutes", type=int, default=90 if q is not t else 120, help="hard stop")
    sp.add_parser("status")
    sp.add_parser("credit")
    sp.add_parser("stop").add_argument("run")
    a = ap.parse_args()
    S.setup()
    if a.cmd == "credit":
        print(f"${credit():.2f}")
    elif a.cmd == "stop":
        destroy(S.group_name(a.run))
    else:
        {"offers": cmd_offers, "gen": cmd_gen, "train": cmd_train, "watch": cmd_watch, "status": cmd_status}[a.cmd](a)


if __name__ == "__main__":
    main()
