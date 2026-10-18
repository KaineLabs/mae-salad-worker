#!/usr/bin/env python3
"""An avatar episode's gen/jobs.json (scripts/avatar_plan.mjs) → a Salad job list, for style.broll "journal":
  talk   her talking shot: the episode's outfit photo + this scene's voice → LTX-2.3 (same prompt as RunPod)
  still  a journal drawing: Z-Image Turbo + the cast LoRAs of the people in it, drawn directly at 1920x1088
  clip   the same drawing as a keyframe (<id>-key.png), then LTX-2.5 animates it: generated at half size and
         doubled by the latent upscaler, so the clip is ~1080p without rendering it at full size
The output names match gen/manifest.json, so `salad.py pull` + this script's --manifest drop them where the
render expects them.

  .venv/bin/python scripts/salad/episode_jobs.py <slug> [--only talk,still,clip] [--ids s02,s05] [--video long]
      → topics/<slug>/<video>/gen/salad.jobs.json   (then: salad.py gen <run> that file)
  .venv/bin/python scripts/salad/episode_jobs.py <slug> --manifest <pulled dir>
      → copies the pulled files into gen/ and records them in gen/manifest.json
"""
import argparse
import json
import os
import re
import shutil
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts", "runpod"))
CAST = json.load(open(os.path.join(ROOT, "avatars", "maes", "cast.json")))
LORA = os.path.join(ROOT, "avatars", "lora")
MOTION = (" Hand-drawn sepia ink and watercolour illustration that stays a drawing: the paper texture and pen lines stay put, "
          "gentle subtle motion, steady camera. No text. Sound: quiet ambience, no speech, no music.")
# Wan 2.2 (picked 2026-10-18 over LTX-2.5 in a side-by-side test: real acting, faces hold) — ask for motion
WAN_MOTION = (" Hand-drawn sepia ink and watercolour illustration that stays a drawing: the paper texture and pen lines stay, "
              "natural expressive motion, the people move and gesture as described, slow gentle camera push-in, no text.")
# the RunPod talking-shot prompt (tested), kept identical
TALK = ("Use <Picture 1> as the first frame and keep the woman's identity, her silver braid, her shawl, the place and the "
        "camera framing exactly. She speaks directly to the camera, saying <Audio 1> exactly as it is, with accurate "
        "lip-sync, natural facial expressions, blinking and small gentle head movements; her hands stay still. "
        "Her manner: {mood}. Static camera, photorealistic, soft natural light, no wind.")


def loras_for(cast):
    """The trained LoRAs of the people in a shot: full strength alone, shared strength when several (less bleed)."""
    people = [c for c in cast if CAST["people"].get(c, {}).get("lora")]
    strength = {1: 1.0, 2: 0.8}.get(len(people), 0.7)
    out = []
    for c in people:
        p = CAST["people"][c]
        out.append({"name": p["lora"], "file": p.get("loraFile", f"{p['lora']}.safetensors"), "strength": strength})
    return out


# Mae's clothes when a prompt doesn't name them: without them she borrows the other person's outfit (Ray's green
# work shirt, Carol's denim jacket in the Vast test, 2026-10-18); the season in the prompt picks one
MAE_WEAR = [(("winter", "christmas", "snow"), "a thick gray wool cardigan over a white blouse"),
            (("autumn", "fall", "thanksgiving", "november"), "a rust-brown knit cardigan over a cream blouse"),
            (("spring", "easter"), "a soft lavender cardigan over a white blouse"),
            (("summer", "reunion", "picnic"), "a light blue cotton blouse with short sleeves")]


def mae_wear(prompt):
    low = prompt.lower()  # the prompts never dress Mae ("wearing …" in them is always someone else's clothes)
    for words, wear in MAE_WEAR:
        if any(w in low for w in words):
            return f", wearing {wear}"
    return ", wearing a cream knit cardigan over a white blouse"


def journal_prompt(prompt, cast):
    """The scene prompt in the journal style, with each LoRA person's trigger right after their name
    ("her son Ray, a man…" → "her son Ray (r4yhnd man), a man…"); Mae, who is "she" in the prompts, is introduced first."""
    lead = ""
    for c in cast:
        p = CAST["people"][c]
        if c == "mae":
            desc = p["description"] + mae_wear(prompt)
            lead = f"{p['trigger']}, {desc}. " if p.get("lora") else f"{desc}. "
            prompt = re.sub(r"^(She|she)\b", "Mae", prompt)
        elif p.get("lora"):
            rel = p["description"].split(",")[0]  # "her son Ray"
            prompt = prompt.replace(rel, f"{rel} ({p['trigger']})", 1)
    if not any(CAST["people"].get(c, {}).get("lora") for c in cast):
        # no cast LoRA (they carry the journal style): objects and hands drift to photographs (s14, s21 in episode 2),
        # so the style goes first, where Z-Image weighs it most
        lead = "A hand-drawn sepia ink and watercolour sketch on aged cream paper, not a photograph, not 3D: " + lead
    return f"{lead}{prompt} {CAST['style']}"


def ref_file(c):
    """A cast member's reference drawing (the same picture their LoRA training set was made from)."""
    for f in (os.path.join(LORA, c, "ref.png"), os.path.join(LORA, c, f"{c}_ref.png")):
        if os.path.exists(f):
            return f
    return None


def ref_sheet(people, out_dir):
    """The people's reference drawings side by side on a 16:9 cream page (Qwen-Image-Edit keeps the input's shape)."""
    from PIL import Image
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "ref-" + "-".join(people) + ".png")
    if not os.path.exists(path):
        W, H = 1664, 936
        page = Image.new("RGB", (W, H), (243, 234, 214))
        n = len(people)
        size = min(H - 40, (W - 40) // n - 20)
        x0 = (W - (n * size + (n - 1) * 20)) // 2
        for i, c in enumerate(people):
            page.paste(Image.open(ref_file(c)).convert("RGB").resize((size, size)), (x0 + i * (size + 20), (H - size) // 2))
        page.save(path)
    return path


def ref_prompt(prompt, cast):
    """Method B (picked 2026-10-18 over Z-Image + LoRAs in a side-by-side test): Qwen-Image-Edit draws the scene from
    the people's reference drawings, so 3-4 people keep their own faces and ages. Clothes are stated per scene, or the
    reference's clothes are copied (Mae's paisley shawl at the lawyer's in the test)."""
    people = [c for c in cast if ref_file(c)]
    who = []
    for c in people:
        p = CAST["people"][c]
        if c == "mae":
            who.append(f"Mae, the elderly woman with the long silver braid, wearing{mae_wear(prompt)[len(', wearing'):]}")
        else:
            who.append(p["description"].replace("Black American ", ""))
    scene = re.sub(r"^(She|she)\b", "Mae", prompt)
    text = (f"Draw a brand-new illustration of this scene with the people from the reference picture, left to right: "
            f"{'; '.join(who)}. Scene: {scene} Keep each person's face, age, skin tone and hair exactly as in the reference, "
            f"dressed as described here, not as in the reference. Exactly the people the scene names, no one else. "
            f"Draw it in the same hand-drawn style: warm sepia-brown ink line work with soft watercolour washes on aged cream "
            f"paper, like an old family journal sketch. No text, no letters. Wide 16:9 composition.")
    return people, text


def fps(path):
    import subprocess
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v", "-show_entries", "stream=r_frame_rate", "-of", "csv=p=0", path],
                       capture_output=True, text=True).stdout.strip() or "24/1"
    n, d = r.split("/")
    return float(n) / float(d or 1)


def finish_wan(src, dest, seconds):
    """A Wan clip (1280x720, 16 fps, ≤ ~7.5 s) → 1920x1080 at 24 fps, motion-interpolated, and slowed just enough to
    cover its line (Remotion plays a clip once; a short clip would end before the voice)."""
    import subprocess
    dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", src],
                               capture_output=True, text=True).stdout.strip() or 0) or 5
    stretch = max(1.0, (seconds + 0.2) / dur)
    vf = f"setpts={stretch:.4f}*PTS,minterpolate=fps=24:mi_mode=mci:mc_mode=aobmc:me_mode=bidir,scale=1920:1080:flags=lanczos"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", src, "-vf", vf, "-an", "-c:v", "libx264", "-crf", "14", "-preset", "slow",
                    "-pix_fmt", "yuv420p", dest], check=True)


CROWD = re.compile(r"\b(adults|relatives|family members|the family|people|guests|everyone|others|cousins|neighbors)\b", re.I)


def lint(sid, prompt, cast):
    """Warnings for prompts that invite strangers or blended faces (episode 2: "the adults around the table" drew
    invented old relatives; four LoRA people in one shot blended faces)."""
    out = []
    named = [c for c in cast if CAST["people"].get(c, {}).get("lora")]
    if CROWD.search(prompt) and not re.search(r"exactly|no one else|nobody else|only", prompt, re.I):
        out.append(f"{sid}: '{CROWD.search(prompt).group(0)}' without saying exactly who: the model invents strangers (say 'exactly N people, no one else' or name them)")
    if len(named) > 4:
        out.append(f"{sid}: {len(named)} recurring people in one shot: even drawn from their references, more than 4 crowd the frame")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("slug")
    ap.add_argument("--video", default="long")
    ap.add_argument("--only", default="talk,still,clip")
    ap.add_argument("--ids")
    ap.add_argument("--manifest")
    ap.add_argument("--stage", choices=["all", "keys", "clips"], default="all",
                    help="keys: stills + every clip's keyframe only (review the sheet, redo bad ones); clips: animate the keyframes of --keys-run")
    ap.add_argument("--method", choices=["ref", "lora"], default="ref",
                    help="ref: drawings with recurring people are made from their reference drawings (Qwen-Image-Edit); lora: Z-Image + cast LoRAs")
    ap.add_argument("--animator", choices=["wan", "ltx"], default="wan", help="clips: Wan 2.2 (default) or LTX-2.5")
    ap.add_argument("--keys-run", help="the run whose out/ holds the reviewed keyframes (for --stage clips)")
    a = ap.parse_args()
    vdir = os.path.join(ROOT, "topics", a.slug, *a.video.split("/"))
    gen = os.path.join(vdir, "gen")
    spec = json.load(open(os.path.join(gen, "jobs.json")))
    scenes = {s["id"]: s for s in json.load(open(os.path.join(vdir, "scenes.json")))["scenes"]}

    if a.manifest:
        mpath = os.path.join(gen, "manifest.json")
        made = json.load(open(mpath)) if os.path.exists(mpath) else {}
        n = 0
        for j in spec["jobs"]:
            src = os.path.join(a.manifest, j["out"])
            if os.path.exists(src):
                dest = os.path.join(gen, j["out"])
                if src.endswith(".mp4") and j["kind"] == "clip" and fps(src) < 20:
                    finish_wan(src, dest, j["seconds"])
                else:
                    shutil.copy(src, dest)
                made[j["id"]] = {"file": j["out"], "kind": j["kind"], "seconds": j["seconds"]}
                n += 1
        json.dump(made, open(mpath, "w"), indent=1)
        print(f"{n} files into {gen} (manifest.json updated)")
        return

    kinds = set(a.only.split(","))
    warn = []
    ids = set(a.ids.split(",")) if a.ids else None
    jobs = []
    for j in spec["jobs"]:
        if j["kind"] not in kinds or (ids and j["id"].replace("-media", "") not in ids):
            continue
        if j["kind"] == "talk":
            jobs.append({"id": j["id"], "kind": "talk", "image": "local:" + j["face"], "audio": "local:" + j["audio"],
                         "seconds": j["seconds"], "size": j["size"], "seed": j["seed"], "prompt": TALK.format(mood=j["mood"]), "out": j["out"]})
            continue
        sc = scenes[j["id"].replace("-media", "")]
        cast = sc["props"].get("cast", [])
        warn += lint(j["id"], j["prompt"], cast)
        still = {"kind": "t2i", "prompt": journal_prompt(j["prompt"], cast), "loras": loras_for(cast), "seed": j["seed"], "width": 1920, "height": 1088}
        people = [c for c in cast if ref_file(c)]
        if a.method == "ref" and people:
            people, text = ref_prompt(j["prompt"], cast)
            sheet = ref_sheet(people, os.path.join(gen, "refs"))
            still = {"kind": "edit", "images": ["local:" + os.path.relpath(sheet, ROOT)], "prompt": text, "seed": j["seed"], "lightning": True}
        if j["kind"] == "still":
            if a.stage != "clips":
                jobs.append({"id": j["id"], **still, "out": j["out"]})
        else:
            key = j["out"].replace(".mp4", "-key.png")
            if a.stage != "clips":
                jobs.append({"id": j["id"] + "-key", **still, "out": key})
            if a.stage != "keys":
                img = f"jobs/{a.keys_run}/out/{key}" if a.stage == "clips" else "out/" + key
                if a.animator == "wan":
                    jobs.append({"id": j["id"], "kind": "wan", "image": img, "prompt": j["prompt"] + WAN_MOTION,
                                 "seconds": j["seconds"], "seed": j["seed"], "out": j["out"]})
                else:
                    jobs.append({"id": j["id"], "kind": "i2v", "image": img, "prompt": j["prompt"] + MOTION,
                                 "seconds": j["seconds"], "seed": j["seed"], "megapixels": 2.1, "out": j["out"]})
    for w in warn:
        print("  warn:", w)
    if a.stage == "clips" and not a.keys_run:
        sys.exit("--stage clips needs --keys-run (the run that made the reviewed keyframes)")
    out = os.path.join(gen, "salad.jobs.json")
    json.dump({"slug": a.slug, "video": a.video, "jobs": jobs}, open(out, "w"), indent=1)
    from collections import Counter
    print(out, dict(Counter(j["kind"] for j in jobs)))


if __name__ == "__main__":
    main()
