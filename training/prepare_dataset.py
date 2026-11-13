"""
Prepare the driver drowsiness dataset with a leakage-free split.

Source: Hugging Face `n7i5x9/driver-drowsiness-dataset`
        23,116 images (640x640), classes drowsy / not_drowsy, unified from
        4 public sources (closed-eye & yawn -> drowsy, open-eye & no-yawn -> not_drowsy).

Face filter
    Despite the source README, roughly 40% of the images are single-eye
    close-ups (e.g. MRL eye crops), not faces. This project monitors a
    driver's whole face from a webcam, so images in which MediaPipe Face
    Landmarker finds no face are excluded (counts are kept in the report).

Why we re-split
    The published splits contain Roboflow augmentation copies of the same
    photo spread across train / valid / test (~67% of test images have an
    identical perceptual hash in train). Evaluating on that test set would
    measure memorisation, not generalisation. We therefore:
      1. group images by their original file stem (the part before ".rf."),
         so all augmented copies of one photo stay together;
      2. assign whole groups to train / valid / test (80 / 10 / 10),
         stratified by class, with a fixed seed;
      3. drop any valid/test image whose 16x16 perceptual hash (dHash) also
         occurs in train (and test images duplicated in valid), removing
         exact / near-exact duplicates that come from different file names.

      4. drop images without a detectable face (see "Face filter").

Output
    datasets/driver_drowsiness/{train,valid,test}/{drowsy,not_drowsy}/*.jpg  (resized to 320x320)
    runs/dataset_report.json

Usage:  python training/prepare_dataset.py [--force]
"""

import argparse
import io
import json
import random
import re
import shutil
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "datasets" / "raw"
OUT = ROOT / "datasets" / "driver_drowsiness"
RUNS = ROOT / "runs"
HF = "https://huggingface.co/datasets/n7i5x9/driver-drowsiness-dataset/resolve/main/"
SHARDS = {
    "train": ["data/train-00000-of-00002.parquet", "data/train-00001-of-00002.parquet"],
    "valid": ["data/validation-00000-of-00001.parquet"],
    "test": ["data/test-00000-of-00001.parquet"],
}
CLASSES = ["drowsy", "not_drowsy"]
SPLIT_FRACTIONS = {"train": 0.8, "valid": 0.1, "test": 0.1}
IMG_SIZE = 320
SEED = 42


def download():
    RAW.mkdir(parents=True, exist_ok=True)
    for files in SHARDS.values():
        for f in files:
            dst = RAW / Path(f).name
            if not dst.exists():
                print(f"Downloading {f} ...")
                urllib.request.urlretrieve(HF + f, dst)


def dhash(img, size=16):
    g = np.asarray(img.convert("L").resize((size + 1, size), Image.BILINEAR), dtype=np.int16)
    return np.packbits(g[:, 1:] > g[:, :-1]).tobytes()


def load_all():
    items = []  # dicts: stem, label, orig_split, hash, jpeg bytes (resized)
    for split, files in SHARDS.items():
        for f in files:
            for row in pq.read_table(RAW / Path(f).name).to_pylist():
                img = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
                path = row["image"]["path"] or ""
                stem = re.sub(r"\.rf\..*$", "", Path(path).name) or f"noname_{len(items)}"
                small = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
                buf = io.BytesIO()
                small.save(buf, "JPEG", quality=95)
                items.append({"stem": stem, "label": row["label"], "orig_split": split,
                              "hash": dhash(img), "bytes": buf.getvalue(), "name": Path(path).stem})
        print(f"loaded {split}: {sum(1 for i in items if i['orig_split'] == split)}")
    return items


def leakage(items, key):
    hashes = defaultdict(set)
    for it in items:
        hashes[it[key]].add(it["hash"])
    out = {}
    for a, b in (("train", "test"), ("train", "valid"), ("valid", "test")):
        ov = hashes[a] & hashes[b]
        out[f"{a}&{b}"] = {"shared_hashes": len(ov), f"pct_of_{b}_unique": round(100 * len(ov) / max(1, len(hashes[b])), 2)}
    return out


def group_split(items):
    groups = defaultdict(list)
    for i, it in enumerate(items):
        groups[it["stem"]].append(i)

    # stratify by the group's majority label
    by_label = defaultdict(list)
    for members in groups.values():
        maj = Counter(items[m]["label"] for m in members).most_common(1)[0][0]
        by_label[maj].append(members)

    rng = random.Random(SEED)
    for label in sorted(by_label):
        glist = by_label[label]
        rng.shuffle(glist)
        total = sum(len(g) for g in glist)
        acc = 0
        bounds = (SPLIT_FRACTIONS["train"] * total, (SPLIT_FRACTIONS["train"] + SPLIT_FRACTIONS["valid"]) * total)
        for members in glist:
            split = "train" if acc < bounds[0] else "valid" if acc < bounds[1] else "test"
            for m in members:
                items[m]["split"] = split
            acc += len(members)

    # remove cross-split duplicates (different file names, same picture)
    train_h = {it["hash"] for it in items if it["split"] == "train"}
    valid_h = {it["hash"] for it in items if it["split"] == "valid" and it["hash"] not in train_h}
    dropped = Counter()
    for it in items:
        if (it["split"] in ("valid", "test") and it["hash"] in train_h) or \
           (it["split"] == "test" and it["hash"] in valid_h):
            dropped[it["split"]] += 1
            it["split"] = None
    return len(groups), dict(dropped)


_det = None


def _init_detector():
    global _det
    import sys
    sys.path.insert(0, str(ROOT / "ml_model"))
    from drowsiness_detector import DrowsinessDetector
    _det = DrowsinessDetector()


def _has_face(jpeg_bytes):
    import cv2
    img = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
    return bool(_det.analyze_bgr(img).get("face_detected"))


def face_filter(items, workers=6):
    """Exclude images with no detectable face (eye-only crops etc.)."""
    from multiprocessing import Pool

    with Pool(workers, initializer=_init_detector) as pool:
        flags = pool.map(_has_face, [it["bytes"] for it in items], chunksize=64)
    excluded = Counter()
    for it, ok in zip(items, flags):
        it["has_face"] = ok
        if not ok and it["split"] is not None:
            excluded[f"{it['split']}/{CLASSES[it['label']]}"] += 1
            it["split"] = None
    return {"images_with_face": int(sum(flags)), "images_without_face": int(len(flags) - sum(flags)),
            "excluded_by_split_class": dict(sorted(excluded.items()))}


def write(items):
    if OUT.exists():
        shutil.rmtree(OUT)
    used = set()
    for it in items:
        if it["split"] is None:
            continue
        d = OUT / it["split"] / CLASSES[it["label"]]
        d.mkdir(parents=True, exist_ok=True)
        name = it["name"]
        n = 1
        while (d, name) in used:
            name = f"{it['name']}_{n}"
            n += 1
        used.add((d, name))
        (d / f"{name}.jpg").write_bytes(it["bytes"])


def main(force):
    download()
    if OUT.exists() and any(OUT.rglob("*.jpg")) and not force:
        print(f"{OUT} already populated; use --force to rebuild.")
        return
    items = load_all()
    n_groups, dropped = group_split(items)
    face_stats = face_filter(items)
    write(items)

    report = {
        "source": "https://huggingface.co/datasets/n7i5x9/driver-drowsiness-dataset",
        "classes": CLASSES,
        "image_size": f"{IMG_SIZE}x{IMG_SIZE} (resized from 640x640)",
        "total_images": len(items),
        "unique_images_by_hash": len({it["hash"] for it in items}),
        "groups": n_groups,
        "dropped_cross_split_duplicates": dropped,
        "face_filter": face_stats,
        "published_split_counts": {s: dict(Counter(CLASSES[i["label"]] for i in items if i["orig_split"] == s)) for s in SHARDS},
        "published_split_leakage": leakage(items, "orig_split"),
        "new_split_counts": {s: dict(Counter(CLASSES[i["label"]] for i in items if i["split"] == s)) for s in SHARDS},
        "new_split_leakage": leakage([i for i in items if i["split"]], "split"),
        "split_method": "group-aware by original file stem, stratified by class, 80/10/10, seed 42; cross-split dHash-16 duplicates removed from valid/test; images without a detectable face excluded",
    }
    RUNS.mkdir(exist_ok=True)
    (RUNS / "dataset_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ("groups", "dropped_cross_split_duplicates", "face_filter", "new_split_counts", "published_split_leakage", "new_split_leakage")}, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    main(ap.parse_args().force)
