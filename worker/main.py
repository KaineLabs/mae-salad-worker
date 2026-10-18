#!/usr/bin/env python3
"""SaladCloud worker for Grandma Mae's episodes. One image, two modes (env MODE):

  train  trains character LoRAs (ostris ai-toolkit, Z-Image Turbo) one after another, for env LORAS="mae,ray,...".
         Everything lives in Cloudflare R2 (bucket R2_BUCKET), so a node that disappears loses nothing:
           lora/<name>/data/        training images + .txt captions (uploaded from the Mac)
           lora/<name>/train.json   trigger, steps, sample prompts (uploaded from the Mac)
           lora/<name>/output/      ai-toolkit's folder (checkpoints + optimizer), mirrored every minute;
                                    on start it is restored and ai-toolkit resumes from the newest checkpoint
           lora/<name>/<name>.safetensors   the finished LoRA (its presence = done; skipped next time)
  gen    runs a job list through ComfyUI: jobs/<RUN>/jobs.json → jobs/<RUN>/out/<file>. A job whose output is
         already in R2 is skipped, so a moved node or a restart just carries on.

Progress goes to status/<RUN or train>.json in R2 every minute. When the work is finished (or MAX_MINUTES passes)
the worker stops its own container group through the Salad API, so nothing keeps billing.
"""
import copy
import glob
import json
import os
import random
import shutil
import subprocess
import sys
import threading
import time
import traceback
import urllib.parse
import urllib.request

import boto3
from botocore.config import Config

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.environ.get("WORK", "/work")
COMFY = "/opt/ComfyUI"
AITK = "/opt/ai-toolkit"
MODE = os.environ.get("MODE", "gen")
RUN = os.environ.get("RUN", "train")
BUCKET = os.environ["R2_BUCKET"]
T0 = time.time()
MAX_MINUTES = float(os.environ.get("MAX_MINUTES", "480"))

s3 = boto3.client("s3", endpoint_url=f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
                  aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"], aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
                  region_name="auto", config=Config(retries={"max_attempts": 10, "mode": "standard"}))
STATUS = {"mode": MODE, "run": RUN, "machine": os.environ.get("SALAD_MACHINE_ID") or os.environ.get("VAST_MACHINE_ID") or "?", "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
          "state": "starting", "log": []}


def log(msg):
    line = "[%s] %s" % (time.strftime("%H:%M:%S"), msg)
    print(line, flush=True)
    STATUS["log"] = (STATUS["log"] + [line])[-40:]


# ---------- R2 ----------

def exists(key):
    try:
        s3.head_object(Bucket=BUCKET, Key=key)
        return True
    except Exception:
        return False


def keys(prefix):
    out, tok = {}, None
    while True:
        r = s3.list_objects_v2(Bucket=BUCKET, Prefix=prefix, **({"ContinuationToken": tok} if tok else {}))
        for o in r.get("Contents", []):
            out[o["Key"]] = o["Size"]
        if not r.get("IsTruncated"):
            return out
        tok = r["NextContinuationToken"]


def get(key, dest):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    s3.download_file(BUCKET, key, dest + ".part")
    os.replace(dest + ".part", dest)
    return dest


def put(path, key):
    s3.upload_file(path, BUCKET, key)


def put_json(obj, key):
    s3.put_object(Bucket=BUCKET, Key=key, Body=json.dumps(obj, indent=1).encode(), ContentType="application/json")


def get_prefix(prefix, dest):
    for k in keys(prefix):
        get(k, os.path.join(dest, os.path.relpath(k, prefix)))


def heartbeat():
    while True:
        STATUS["updated"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        STATUS["minutes"] = round((time.time() - T0) / 60, 1)
        try:
            put_json(STATUS, f"status/{RUN}.json")
        except Exception as e:
            print("status upload failed:", e, flush=True)
        if time.time() - T0 > MAX_MINUTES * 60:
            log("MAX_MINUTES reached: stopping the container group")
            stop_group()
        time.sleep(60)


# ---------- Salad / Vast.ai ----------

# On Vast.ai every instance carries CONTAINER_ID and CONTAINER_API_KEY, a key that can only stop or destroy that
# one instance: the worker destroys itself when done (destroyed = no GPU, disk or bandwidth billing).
VAST_ID, VAST_KEY = os.environ.get("CONTAINER_ID"), os.environ.get("CONTAINER_API_KEY")


def vast_destroy():
    for _ in range(5):
        try:
            urllib.request.urlopen(urllib.request.Request(f"https://console.vast.ai/api/v0/instances/{VAST_ID}/", method="DELETE",
                                                          headers={"Authorization": f"Bearer {VAST_KEY}"}), timeout=30)
            log("destroy requested")
            return
        except Exception as e:
            log(f"destroy failed: {e}")
            time.sleep(10)


def stop_group():
    """Stops our own container group (billing ends); the Mac tool also stops it when it sees the run is done."""
    if VAST_ID and VAST_KEY:
        return vast_destroy()
    org, proj, name, key = (os.environ.get(k) for k in ("SALAD_ORG", "SALAD_PROJECT", "SALAD_GROUP", "SALAD_API_KEY"))
    if not all((org, proj, name, key)):
        log("no Salad credentials: cannot stop the group myself")
        return
    url = f"https://api.salad.com/api/public/organizations/{org}/projects/{proj}/containers/{name}/stop"
    for _ in range(5):
        try:
            urllib.request.urlopen(urllib.request.Request(url, method="POST", headers={"Salad-Api-Key": key, "User-Agent": "mae-salad/1.0"}), timeout=30)
            log("stop requested")
            return
        except Exception as e:
            log(f"stop failed: {e}")
            time.sleep(10)


def reallocate_self(why):
    """Asks Salad to move this run to another machine (a broken or overloaded node: 60x slower edits were seen).
    On Vast.ai the instance records the machine as bad and destroys itself; the Mac's `vast.py watch` rents another."""
    if VAST_ID and VAST_KEY:
        log(f"too slow ({why}): leaving this machine")
        STATUS["state"] = "slow machine"
        STATUS["bad_machine"] = STATUS["machine"]
        put_json(STATUS, f"status/{RUN}.json")
        vast_destroy()
        time.sleep(3600)
        return
    org, proj, name, key = (os.environ.get(k) for k in ("SALAD_ORG", "SALAD_PROJECT", "SALAD_GROUP", "SALAD_API_KEY"))
    me = os.environ.get("SALAD_MACHINE_ID")
    base = f"https://api.salad.com/api/public/organizations/{org}/projects/{proj}/containers/{name}/instances"
    h = {"Salad-Api-Key": key or "", "User-Agent": "mae-salad/1.0"}
    try:
        with urllib.request.urlopen(urllib.request.Request(base, headers=h), timeout=30) as r:
            inst = [i for i in json.loads(r.read())["instances"] if i.get("machine_id") == me]
        if inst:
            log(f"too slow ({why}): asking Salad for another machine")
            put_json(STATUS, f"status/{RUN}.json")
            urllib.request.urlopen(urllib.request.Request(f"{base}/{inst[0]['id']}/reallocate", method="POST", headers=h), timeout=30)
            time.sleep(600)
    except Exception as e:
        log(f"reallocate failed: {e}")


# seconds a job may take on a healthy 24 GB card before the machine counts as broken (warm-up job excluded)
SLOW = {"edit": 120, "t2i": 120, "i2v": 900, "talk": 1500, "wan": 1500, "h3": 2400}


def finish(state):
    STATUS["state"] = state
    STATUS["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    put_json(STATUS, f"status/{RUN}.json")
    stop_group()
    time.sleep(3600)  # a container that exits is restarted on another node; wait for the stop instead


# ---------- models ----------

MODELS = json.load(open(os.path.join(HERE, "models.json")))


def download_models(groups):
    from concurrent.futures import ThreadPoolExecutor
    from huggingface_hub import hf_hub_download
    items = [it for g in groups for it in MODELS[g]]

    def one(it):
        repo, path, d = it
        dest = os.path.join(COMFY, "models", d, os.path.basename(path))
        if os.path.exists(dest):
            return
        t = time.time()
        # HuggingFace's storage (xet) drops a download now and then: retry before giving up on the machine
        for attempt in range(4):
            try:
                p = hf_hub_download(repo, path, local_dir=os.path.join(WORK, "dl"), token=os.environ.get("HF_TOKEN") or None)
                break
            except Exception as e:
                if attempt == 3:
                    raise
                log(f"download {os.path.basename(path)} failed ({str(e)[:120]}); retrying")
                time.sleep(20 * (attempt + 1))
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.move(p, dest)
        log("model %s %.1f GB in %.0f s" % (os.path.basename(path), os.path.getsize(dest) / 1e9, time.time() - t))

    with ThreadPoolExecutor(6) as ex:
        list(ex.map(one, items))


# ---------- train ----------

def mirror(src, prefix, stop):
    """Uploads new or changed files under src to R2 every minute (and once more when stop is set)."""
    sent = {}
    while True:
        last = stop.is_set()
        for p in glob.glob(os.path.join(src, "**", "*"), recursive=True):
            if not os.path.isfile(p) or p.endswith((".part", ".tmp")):
                continue
            st = os.stat(p)
            if time.time() - st.st_mtime < 20 and not last:
                continue  # still being written
            sig = (st.st_size, st.st_mtime)
            if sent.get(p) != sig:
                try:
                    put(p, prefix + os.path.relpath(p, src))
                    sent[p] = sig
                except Exception as e:
                    print("upload failed", p, e, flush=True)
        if last:
            return
        stop.wait(60)


def train_config(name, spec, data, out):
    trig = spec["trigger"]
    return {
        "job": "extension",
        "config": {
            "name": name,
            "process": [{
                "type": "sd_trainer",
                "training_folder": out,
                "device": "cuda:0",
                "trigger_word": trig,
                "network": {"type": "lora", "linear": spec.get("rank", 16), "linear_alpha": spec.get("rank", 16)},
                "save": {"dtype": "float16", "save_every": spec.get("save_every", 100), "max_step_saves_to_keep": 3},
                "datasets": [{"folder_path": data, "caption_ext": "txt", "caption_dropout_rate": 0.05, "shuffle_tokens": False,
                              "cache_latents_to_disk": True, "resolution": spec.get("resolution", [512, 768, 1024])}],
                "train": {"batch_size": 1, "steps": spec.get("steps", 2500), "gradient_accumulation": 1, "train_unet": True,
                          "train_text_encoder": False, "gradient_checkpointing": True, "noise_scheduler": "flowmatch",
                          "timestep_type": "weighted", "optimizer": "adamw8bit", "lr": spec.get("lr", 1e-4), "dtype": "bf16",
                          "unload_text_encoder": False},
                "model": {"name_or_path": "Tongyi-MAI/Z-Image-Turbo", "arch": "zimage:turbo",
                          "assistant_lora_path": "ostris/zimage_turbo_training_adapter/zimage_turbo_training_adapter_v2.safetensors",
                          "quantize": True, "qtype": "qfloat8", "quantize_te": True, "qtype_te": "qfloat8", "low_vram": False},
                "sample": {"sampler": "flowmatch", "sample_every": spec.get("sample_every", 500), "width": 1024, "height": 576,
                           "prompts": spec.get("samples", []), "neg": "", "seed": 42, "walk_seed": False,
                           "guidance_scale": 1, "sample_steps": 9},
            }],
        },
        "meta": {"name": "[name]", "version": "1.0"},
    }


def run_train():
    names = [n.strip() for n in os.environ.get("LORAS", "").split(",") if n.strip()]
    STATUS["loras"] = {n: "waiting" for n in names}
    for name in names:
        final = f"lora/{name}/{name}.safetensors"
        if exists(final):
            STATUS["loras"][name] = "done"
            log(f"{name}: already trained, skipped")
            continue
        STATUS["state"] = f"training {name}"
        STATUS["loras"][name] = "preparing"
        root = os.path.join(WORK, "lora", name)
        data, out = os.path.join(root, "data"), os.path.join(root, "output")
        get_prefix(f"lora/{name}/data/", data)
        spec = json.load(open(get(f"lora/{name}/train.json", os.path.join(root, "train.json"))))
        get_prefix(f"lora/{name}/output/", out)  # resume point (ai-toolkit picks the newest step file + optimizer.pt)
        have = sorted(glob.glob(os.path.join(out, name, f"{name}_*.safetensors")))
        log(f"{name}: {len(glob.glob(os.path.join(data, '*.txt')))} captions, resuming from {os.path.basename(have[-1]) if have else 'scratch'}")
        import yaml
        cfg = os.path.join(root, "config.yaml")
        yaml.safe_dump(train_config(name, spec, data, out), open(cfg, "w"), sort_keys=False)
        stop = threading.Event()
        th = threading.Thread(target=mirror, args=(out, f"lora/{name}/output/", stop), daemon=True)
        th.start()
        STATUS["loras"][name] = "training"
        logf = open(os.path.join(root, "train.log"), "a")
        p = subprocess.Popen(["/opt/aitk-venv/bin/python", "run.py", cfg], cwd=AITK, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in p.stdout:
            logf.write(line)
            line = line.rstrip()
            if line and ("it/s" in line or "s/it" in line):
                STATUS["progress"] = line[-160:]  # tqdm line: step and speed
            elif line:
                print(line, flush=True)
        rc = p.wait()
        stop.set()
        th.join()
        logf.close()
        put(os.path.join(root, "train.log"), f"lora/{name}/train-{int(time.time())}.log")
        done = os.path.join(out, name, f"{name}.safetensors")
        if rc != 0 or not os.path.exists(done):
            STATUS["loras"][name] = f"failed (exit {rc})"
            log(f"{name}: training failed, exit {rc}")
            continue
        put(done, final)
        STATUS["loras"][name] = "done"
        log(f"{name}: finished → {final}")
    failed = [n for n, s in STATUS["loras"].items() if s != "done"]
    finish("done" if not failed else "failed: " + ",".join(failed))


# ---------- gen (ComfyUI) ----------

def wf(name):
    return copy.deepcopy(json.load(open(os.path.join(HERE, "workflows", name + ".api.json"))))


def comfy(path, body=None, timeout=60):
    req = urllib.request.Request("http://127.0.0.1:8188" + path, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read() or b"{}")


def run_workflow(w, timeout):
    pid = comfy("/prompt", {"prompt": w, "client_id": "salad"})["prompt_id"]
    t = time.time()
    while time.time() - t < timeout:
        h = comfy("/history/" + pid)
        if pid in h:
            st = h[pid].get("status", {})
            if st.get("status_str") == "error":
                raise RuntimeError(json.dumps(st.get("messages", []))[-800:])
            if st.get("completed"):
                return h[pid]
        time.sleep(2)
    comfy("/interrupt", {})
    raise TimeoutError(f"job took over {timeout:.0f} s")


def output_file(hist, exts):
    for o in hist["outputs"].values():
        for k in ("images", "videos", "gifs", "audio"):
            for f in o.get(k, []):
                if f["filename"].lower().endswith(exts):
                    return os.path.join(COMFY, "output", f.get("subfolder", ""), f["filename"])
    raise RuntimeError("no output file")


def input_file(key):
    """An R2 key (inputs/... or out/... of this run, or any full key) → a file in ComfyUI/input, returns its name."""
    full = key if "/" in key and not key.startswith(("inputs/", "out/")) else f"jobs/{RUN}/{key}"
    name = full.replace("/", "_")
    dest = os.path.join(COMFY, "input", name)
    if not os.path.exists(dest):
        get(full, dest)
    return name


def lora_file(spec):
    """A LoRA from R2 (lora/<name>/<file>) into ComfyUI/models/loras."""
    name = spec["name"]
    fname = spec.get("file", f"{name}.safetensors")
    dest = os.path.join(COMFY, "models", "loras", f"{name}__{fname}")
    if not os.path.exists(dest):
        get(f"lora/{name}/{fname}", dest)
    return os.path.basename(dest)


def job_t2i(j):
    w = wf("image_zimage_turbo_lora")
    model = ["4", 0]
    for i, l in enumerate(j.get("loras", [])):
        nid = f"20{i}"
        w[nid] = {"class_type": "LoraLoaderModelOnly", "inputs": {"lora_name": lora_file(l), "strength_model": l.get("strength", 1.0), "model": model}}
        model = [nid, 0]
    w["8"]["inputs"]["model"] = model
    w["5"]["inputs"]["text"] = j["prompt"]
    w["7"]["inputs"].update({"width": j.get("width", 1920), "height": j.get("height", 1088)})
    w["8"]["inputs"].update({"seed": j["seed"], "steps": j.get("steps", 9)})
    return output_file(run_workflow(w, 600), (".png",))


def job_edit(j):
    w = wf("image_qwen_image_edit_2511_int8")
    w["41"]["inputs"]["image"] = input_file(j["images"][0])
    w["170:151"]["inputs"]["prompt"] = j["prompt"]
    w["170:168"]["inputs"]["value"] = bool(j.get("lightning", True))  # 4-step Lightning LoRA (fast) or 40 steps
    w["170:169"]["inputs"]["seed"] = j["seed"]
    w["195"]["inputs"]["filename_prefix"] = "edit"
    return output_file(run_workflow(w, 900), (".png",))


def job_i2v(j):
    w = wf("video_ltx2_5_i2v")
    w["395"]["inputs"]["image"] = input_file(j["image"])
    w["398:376"]["inputs"]["value"] = j["prompt"]
    w["398:383"]["inputs"]["value"] = False  # our prompt, no rewrite
    w["398:362"]["inputs"]["value"] = int(j["seconds"])
    w["398:339"]["inputs"]["noise_seed"] = j["seed"]
    # generated at half size, then the 2x latent upscaler (megapixels = the final size)
    w["403"]["inputs"]["megapixels"] = j.get("megapixels", 2.1)
    return output_file(run_workflow(w, max(900, j["seconds"] * 200)), (".mp4",))


def job_wan(j):
    """Wan 2.2 image-to-video (14B, two experts, 4-step lightx2v LoRAs): 1280x720 at 16 fps, 81 frames = ~5 s
    (frames: 4n+1; longer clips go up to 121 = ~7.5 s)."""
    w = wf("video_wan2_2_i2v")
    w["11"]["inputs"]["image"] = input_file(j["image"])
    w["9"]["inputs"]["text"] = j["prompt"]
    w["12"]["inputs"]["length"] = min(121, max(33, int(j.get("seconds", 5) * 16) // 4 * 4 + 1))
    w["13"]["inputs"]["noise_seed"] = j["seed"]
    # small and fast (e.g. 832x480, ~2x quicker than 720p), then a 2x line-art upscaler on the GPU
    w["12"]["inputs"]["width"], w["12"]["inputs"]["height"] = j.get("width", 1280), j.get("height", 720)
    if j.get("upscaler"):
        w["18"] = {"class_type": "UpscaleModelLoader", "inputs": {"model_name": j["upscaler"]}}
        w["19"] = {"class_type": "ImageUpscaleWithModel", "inputs": {"upscale_model": ["18", 0], "image": ["15", 0]}}
        w["16"]["inputs"]["images"] = ["19", 0]
    return output_file(run_workflow(w, max(900, j.get("seconds", 5) * 240)), (".mp4",))


def job_h3(j):
    """MiniMax H3 image-to-video (33B, pruned int8 + 8-step turbo LoRA, NVFP4 text encoder: needs a Blackwell card,
    RTX 5090 / PRO 6000): 1344x768 at 24 fps; frames on its 17k+5 grid (5 s = 124)."""
    w = wf("video_minimax_h3_i2v")
    if j.get("image"):
        w["114"]["inputs"]["image"] = input_file(j["image"])
    else:
        # text only (t2va): H3 draws the scene itself
        del w["114"]
        del w["104"]["inputs"]["first_frame"]
    w["104"]["inputs"]["prompt"] = j["prompt"]
    n = max(5, round(j.get("seconds", 5) * 24))
    w["104"]["inputs"]["length"] = n + (5 - n % 17) % 17
    w["15"]["inputs"]["noise_seed"] = j["seed"]
    return output_file(run_workflow(w, max(1200, j.get("seconds", 5) * 300)), (".mp4",))


def job_talk(j):
    w = wf("video_ltx2_3_ia2v")
    w["269"]["inputs"]["image"] = input_file(j["image"])
    w["276"]["inputs"]["audio"] = input_file(j["audio"])
    w["340:319"]["inputs"]["value"] = j["prompt"]
    w["340:349"]["inputs"]["value"] = False
    w["340:331"]["inputs"]["value"] = round(j["seconds"], 2)
    w["340:330"]["inputs"]["value"], w["340:324"]["inputs"]["value"] = j.get("size", [1920, 1088])
    w["340:285"]["inputs"]["noise_seed"] = j["seed"]
    return output_file(run_workflow(w, max(1200, j["seconds"] * 120)), (".mp4",))


KINDS = {"t2i": (job_t2i, ["zimage"]), "edit": (job_edit, ["qwen_edit"]), "i2v": (job_i2v, ["ltx25"]), "talk": (job_talk, ["ltx23"]), "wan": (job_wan, ["wan22", "upscale"]), "h3": (job_h3, ["h3"])}


def run_gen():
    spec = json.load(open(get(f"jobs/{RUN}/jobs.json", os.path.join(WORK, "jobs.json"))))
    done = set(keys(f"jobs/{RUN}/out/"))
    todo = [j for j in spec["jobs"] if f"jobs/{RUN}/out/{j['out']}" not in done]
    STATUS.update(total=len(spec["jobs"]), done=len(spec["jobs"]) - len(todo), failed=[])
    log(f"{len(todo)} of {len(spec['jobs'])} jobs to do")
    if not todo:
        finish("done")
    groups = sorted({g for j in todo for g in KINDS[j["kind"]][1]})
    STATUS["state"] = "downloading models: " + ", ".join(groups)
    download_models(groups)
    STATUS["state"] = "starting ComfyUI"
    subprocess.Popen(["/opt/comfy-venv/bin/python", "main.py", "--listen", "127.0.0.1", "--port", "8188"], cwd=COMFY,
                     stdout=open(os.path.join(WORK, "comfy.log"), "a"), stderr=subprocess.STDOUT)
    for _ in range(300):
        try:
            comfy("/system_stats")
            break
        except Exception:
            time.sleep(2)
    else:
        raise RuntimeError("ComfyUI did not start")
    STATUS["state"] = "generating"
    timings = {}
    # stills before the clips made from them (a clip's image can be another job's output)
    order = {"edit": 0, "t2i": 1, "talk": 2, "i2v": 3, "wan": 4, "h3": 5}
    for j in sorted(todo, key=lambda j: order[j["kind"]]):
        t = time.time()
        STATUS["current"] = f"{j['kind']} {j['id']}"
        try:
            f = KINDS[j["kind"]][0](j)
            put(f, f"jobs/{RUN}/out/{j['out']}")
            os.remove(f)
            STATUS["done"] += 1
            timings[j["id"]] = round(time.time() - t, 1)
            log(f"{j['kind']} {j['id']} done in {timings[j['id']]:.0f} s ({STATUS['done']}/{STATUS['total']})")
            # full-quality edits (40 steps, no Lightning LoRA) take ~6x longer than the 4-step ones the limit assumes
            slow = SLOW[j["kind"]] * (6 if j["kind"] == "edit" and not j.get("lightning", True) else 1)
            if len(timings) > 1 and timings[j["id"]] > slow * (j.get("seconds", 1) if j["kind"] in ("i2v", "talk") else 1) / (10 if j["kind"] in ("i2v", "talk") else 1):
                reallocate_self(f"{j['id']} took {timings[j['id']]:.0f} s")
        except TimeoutError as e:
            STATUS["failed"].append(j["id"])
            log(f"{j['kind']} {j['id']} FAILED: {str(e)[:300]}")
            reallocate_self(f"{j['id']} timed out")
        except Exception as e:
            STATUS["failed"].append(j["id"])
            log(f"{j['kind']} {j['id']} FAILED: {str(e)[:300]}")
        put_json(timings, f"jobs/{RUN}/timings-{STATUS['machine']}.json")
    STATUS.pop("current", None)
    finish("done" if not STATUS["failed"] else "done with failures")


if __name__ == "__main__":
    os.makedirs(WORK, exist_ok=True)
    threading.Thread(target=heartbeat, daemon=True).start()
    try:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
        STATUS["gpu"] = gpu
        log("GPU: " + gpu)
        ok = subprocess.run(["/opt/comfy-venv/bin/python", "-c", "import torch; assert torch.cuda.is_available(); torch.zeros(1).cuda()"]).returncode == 0
        if not ok and VAST_ID:
            reallocate_self("CUDA not usable on this machine")
        if not ok:
            # CUDA 13 wheels need driver >= 580; exiting makes Salad restart us, and repeated failures move us to another node
            log("CUDA not usable on this node (driver too old?): exiting so Salad reallocates")
            put_json(STATUS, f"status/{RUN}.json")
            sys.exit(1)
        run_train() if MODE == "train" else run_gen()
    except Exception:
        log("CRASH: " + traceback.format_exc()[-1500:])
        finish("crashed")
