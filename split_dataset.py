import os, shutil, random
from pathlib import Path

random.seed(42)

SRC = Path("data/labeled")

images = sorted([f for f in SRC.glob("*.jpg")])
random.shuffle(images)

n = len(images)
cut1 = int(n * 0.70)
cut2 = int(n * 0.85)

split_map = {
    "train": images[:cut1],
    "val":   images[cut1:cut2],
    "test":  images[cut2:],
}

for split, files in split_map.items():
    img_dir = SRC / split / "images"
    lbl_dir = SRC / split / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    for img_path in files:
        lbl_path = SRC / (img_path.stem + ".txt")
        shutil.copy(img_path, img_dir / img_path.name)
        if lbl_path.exists():
            shutil.copy(lbl_path, lbl_dir / lbl_path.name)

    print(f"{split}: {len(files)} images copied")

print("\nDone! Split complete.")