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
    return f"{lead}{prompt} {CAST['style']}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("slug")
    ap.add_argument("--video", default="long")
    ap.add_argument("--only", default="talk,still,clip")
    ap.add_argument("--ids")
    ap.add_argument("--manifest")
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
                shutil.copy(src, os.path.join(gen, j["out"]))
                made[j["id"]] = {"file": j["out"], "kind": j["kind"], "seconds": j["seconds"]}
                n += 1
        json.dump(made, open(mpath, "w"), indent=1)
        print(f"{n} files into {gen} (manifest.json updated)")
        return

    kinds = set(a.only.split(","))
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
        still = {"kind": "t2i", "prompt": journal_prompt(j["prompt"], cast), "loras": loras_for(cast), "seed": j["seed"], "width": 1920, "height": 1088}
        if j["kind"] == "still":
            jobs.append({"id": j["id"], **still, "out": j["out"]})
        else:
            key = j["out"].replace(".mp4", "-key.png")
            jobs.append({"id": j["id"] + "-key", **still, "out": key})
            jobs.append({"id": j["id"], "kind": "i2v", "image": "out/" + key, "prompt": j["prompt"] + MOTION,
                         "seconds": j["seconds"], "seed": j["seed"], "megapixels": 2.1, "out": j["out"]})
    out = os.path.join(gen, "salad.jobs.json")
    json.dump({"slug": a.slug, "video": a.video, "jobs": jobs}, open(out, "w"), indent=1)
    from collections import Counter
    print(out, dict(Counter(j["kind"] for j in jobs)))


if __name__ == "__main__":
    main()
