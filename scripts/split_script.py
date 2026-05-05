import os
import shutil
import random
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
labeled_dir = Path(r"C:\Users\vagee\OneDrive\Desktop\major project\pcb-defect-detection\data\labeled")
output_dir  = Path(r"C:\Users\vagee\OneDrive\Desktop\major project\pcb-defect-detection\data\split")

# ── Split ratios ───────────────────────────────────────────────────────────
TRAIN_RATIO = 0.80
VAL_RATIO   = 0.10
TEST_RATIO  = 0.10

# ── Create output folders ──────────────────────────────────────────────────
for split in ["train", "val", "test"]:
    (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
    (output_dir / split / "labels").mkdir(parents=True, exist_ok=True)

# ── Collect all image stems (e.g. "92000000_test") ────────────────────────
image_stems = [f.stem for f in labeled_dir.glob("*.jpg")]

if not image_stems:
    print("❌ No images found in labeled_dir. Check the path.")
    exit()

# ── Shuffle for randomness ─────────────────────────────────────────────────
random.seed(42)          # fixed seed → reproducible split every run
random.shuffle(image_stems)

# ── Calculate split sizes ──────────────────────────────────────────────────
total      = len(image_stems)
train_end  = int(total * TRAIN_RATIO)
val_end    = train_end + int(total * VAL_RATIO)

splits = {
    "train": image_stems[:train_end],
    "val":   image_stems[train_end:val_end],
    "test":  image_stems[val_end:],
}

# ── Copy files ────────────────────────────────────────────────────────────
counts = {"train": 0, "val": 0, "test": 0}
skipped = 0

for split_name, stems in splits.items():
    for stem in stems:
        img_src = labeled_dir / f"{stem}.jpg"
        lbl_src = labeled_dir / f"{stem}.txt"

        if not img_src.exists() or not lbl_src.exists():
            skipped += 1
            continue

        shutil.copy(img_src, output_dir / split_name / "images" / f"{stem}.jpg")
        shutil.copy(lbl_src, output_dir / split_name / "labels" / f"{stem}.txt")
        counts[split_name] += 1

# ── Summary ───────────────────────────────────────────────────────────────
print(f"✅ Split complete!")
print(f"   Train : {counts['train']} images")
print(f"   Val   : {counts['val']}   images")
print(f"   Test  : {counts['test']}  images")
if skipped:
    print(f"   ⚠️  Skipped {skipped} items (missing image or label)")