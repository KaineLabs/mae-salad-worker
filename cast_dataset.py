#!/usr/bin/env python3
"""Training sets for the cast LoRAs (Mae's journal-style memories), made on Salad with Qwen-Image-Edit-2511.

  .venv/bin/python scripts/salad/cast_dataset.py jobs mae,ray,david,carol   → avatars/lora/datasets.jobs.json (then: salad.py gen datasets …)
  .venv/bin/python scripts/salad/cast_dataset.py sheet <dir>                 → a contact sheet of the results to review
  .venv/bin/python scripts/salad/cast_dataset.py data <name> <dir> [--drop 05,11]
        → avatars/lora/<name>/data/ (00 = the reference + the kept images, captions with the trigger) and train.json

Each person: avatars/lora/<name>/ref.png (the chosen reference drawing) and their entry in avatars/maes/cast.json.
Same recipe as Harold's LoRA (avatars/lora/harold), which kept his face across scenes: one reference, 16 edits
(angles, expressions, full body, sitting, hands busy), the drifted ones dropped by eye.
"""
import json
import os
import shutil
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
LORA = os.path.join(ROOT, "avatars", "lora")
CAST = json.load(open(os.path.join(ROOT, "avatars", "maes", "cast.json")))
TAIL = "Hand-drawn sepia ink and watercolour illustration on aged cream paper, old family journal sketch."

# id, caption (what the picture is), edit instruction ({p} = him/her, {s} = he/she, {o} = his/her)
SHOTS = [
    ("01", "close-up head-and-shoulders, front view, warm open smile", "close-up portrait of {p} facing the viewer with a warm open smile"),
    ("02", "close-up three-quarter view, worried expression", "close-up of {o} head and shoulders turned three-quarters to the left, brow furrowed, a worried expression"),
    ("03", "close-up profile view, calm expression", "close-up of {o} head and shoulders in side profile facing right, calm and thoughtful"),
    ("04", "head-and-shoulders, laughing with head tilted back", "head and shoulders, laughing heartily with {o} head tilted back a little and eyes crinkled"),
    ("05", "medium shot, tired, rubbing eyes", "waist-up, looking tired, rubbing {o} eyes with one hand, shoulders slumped"),
    ("06", "medium shot, sitting at a wooden kitchen table with a cup of coffee", "sitting at a simple wooden kitchen table holding a cup of coffee in both hands, a relaxed small smile, a plain wall lightly washed in ochre behind"),
    ("07", "medium shot, sitting at a table looking down at papers", "sitting at a table looking down at a few plain sheets of paper with no writing, serious and thoughtful"),
    ("08", "full body, standing, three-quarter view", "full body, standing relaxed, body turned three-quarters, a faint ground shadow"),
    ("09", "full body, walking toward the viewer", "full body, walking toward the viewer mid-stride, a faint ground shadow"),
    ("10", "medium shot, arms crossed, serious", "waist-up, arms crossed, a serious, guarded expression"),
    ("11", "full body, sitting on a wooden porch step, elbows on knees, pensive", "full body, sitting on a wooden porch step with elbows on knees, looking off into the distance, pensive, a simple porch suggested with a few lines"),
    ("12", "medium shot, back three-quarter view looking over the shoulder", "waist-up, seen from behind three-quarters, looking back over {o} shoulder with a gentle smile"),
    ("13", "medium shot, talking on a phone", "waist-up, holding a phone to {o} ear, listening with a concerned look"),
    ("14", "head-and-shoulders, low angle, looking down and smiling", "head and shoulders seen from a slightly low angle, looking down toward the viewer and smiling kindly"),
    ("15", "medium shot, hugging someone seen from behind", "waist-up, hugging a person whose back is to the viewer, {o} eyes closed, tender"),
    ("16", "medium shot, surprised expression, eyebrows raised", "waist-up, surprised, eyebrows raised and mouth slightly open, one hand raised"),
]
# Mae's memories span seasons: vary her clothes so the LoRA learns her face, not the shawl
MAE_CLOTHES = {"05": "a pale blue cardigan over a white blouse", "06": "a pale blue cardigan over a white blouse", "07": "a soft green sweater",
               "09": "a long gray wool winter coat", "11": "a plain cream sweater", "13": "a soft green sweater", "15": "a flowered apron over a white blouse",
               "16": "a plain cream sweater"}
PRON = {"mae": ("her", "she", "her"), "carol": ("her", "she", "her")}


def clothes(name, sid):
    if name == "mae":
        return MAE_CLOTHES.get(sid, "a rust paisley shawl over a white blouse")
    return {"ray": "a faded green work shirt", "david": "a navy crew-neck sweater", "carol": "a denim jacket over a white top"}[name]


def jobs(names):
    out = []
    for n in names:
        p, s, o = PRON.get(n, ("him", "he", "his"))
        for sid, cap, act in SHOTS:
            wear = clothes(n, sid)
            prompt = (f"Draw this exact same person again in a new picture: {act.format(p=p, s=s, o=o)}, wearing {wear}, "
                      f"plain cream paper background unless a setting is named. Keep the same face, age, skin tone, hair and features exactly, "
                      f"and the same hand-drawn style: warm sepia-brown ink line work with soft watercolour washes on aged cream paper, "
                      f"like an old family journal sketch. Only this one person, no text.")
            out.append({"id": f"{n}-{sid}", "kind": "edit", "images": [f"local:avatars/lora/{n}/ref.png"], "prompt": prompt,
                        "seed": 7, "lightning": True, "out": f"{n}/{sid}.png"})
    return {"jobs": out}


def caption(name, sid, cap):
    c = CAST["people"][name]
    return f"{c['trigger']}, {cap}, wearing {clothes(name, sid)}. {TAIL}\n"


def data(name, src, drop):
    d = os.path.join(LORA, name, "data")
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    shutil.copy(os.path.join(LORA, name, "ref.png"), os.path.join(d, "00.png"))
    ref_cap = "waist-up portrait, front view, calm slight smile"
    open(os.path.join(d, "00.txt"), "w").write(caption(name, "00", ref_cap))
    kept = 0
    for sid, cap, _ in SHOTS:
        f = os.path.join(src, f"{sid}.png")
        if sid in drop or not os.path.exists(f):
            continue
        shutil.copy(f, os.path.join(d, f"{sid}.png"))
        open(os.path.join(d, f"{sid}.txt"), "w").write(caption(name, sid, cap))
        kept += 1
    c = CAST["people"][name]
    trig = c["trigger"]
    train = {"trigger": trig, "steps": 1200, "rank": 16, "lr": 1e-4, "save_every": 100, "sample_every": 400,
             "samples": [f"{trig}, sitting at a farmhouse kitchen table at night with a cup of coffee, warm lamp light. {TAIL}",
                         f"{trig}, full body, standing on a farmhouse porch in autumn, smiling. {TAIL}"]}
    json.dump(train, open(os.path.join(LORA, name, "train.json"), "w"), indent=1)
    print(f"{name}: reference + {kept} images → {d}; train.json ({train['steps']} steps)")


def sheet(src):
    from PIL import Image, ImageDraw
    files = sorted(f for f in os.listdir(src) if f.endswith(".png"))
    cols, size = 4, 384
    rows = (len(files) + cols - 1) // cols
    out = Image.new("RGB", (cols * size, rows * size), "white")
    for i, f in enumerate(files):
        im = Image.open(os.path.join(src, f)).convert("RGB")
        im.thumbnail((size, size))
        x, y = (i % cols) * size, (i // cols) * size
        out.paste(im, (x, y))
        ImageDraw.Draw(out).text((x + 8, y + 8), f[:-4], fill="red")
    path = os.path.join(src, "sheet.jpg")
    out.save(path, quality=88)
    print(path, f"({rows} rows)")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "jobs":
        p = os.path.join(LORA, "datasets.jobs.json")
        json.dump(jobs(sys.argv[2].split(",")), open(p, "w"), indent=1)
        print(p)
    elif cmd == "sheet":
        sheet(sys.argv[2])
    elif cmd == "data":
        drop = sys.argv[sys.argv.index("--drop") + 1].split(",") if "--drop" in sys.argv else []
        data(sys.argv[2], sys.argv[3], drop)
