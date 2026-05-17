import cv2
import random
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
labeled_dir = Path(r"C:\Users\vagee\OneDrive\Desktop\major project\pcb-defect-detection\data\labeled")

# ── Class names and colors (BGR) ───────────────────────────────────────────
CLASSES = {
    0: ("missing_hole",    (255, 0,   0  )),  # blue
    1: ("mouse_bite",      (0,   255, 0  )),  # green
    2: ("open_circuit",    (0,   0,   255)),  # red
    3: ("short",           (255, 255, 0  )),  # cyan
    4: ("spur",            (0,   255, 255)),  # yellow
    5: ("spurious_copper", (255, 0,   255)),  # magenta
}

# ── How many random images to show ────────────────────────────────────────
NUM_IMAGES = 5

# ── Collect all images ────────────────────────────────────────────────────
all_images = list(labeled_dir.glob("*.jpg"))

if not all_images:
    print("❌ No images found. Check the labeled_dir path.")
    exit()

# Pick random samples
samples = random.sample(all_images, min(NUM_IMAGES, len(all_images)))

for img_path in samples:
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"⚠️  Could not read {img_path.name}")
        continue

    h, w = img.shape[:2]

    # Load corresponding label file
    lbl_path = labeled_dir / f"{img_path.stem}.txt"
    if not lbl_path.exists():
        print(f"⚠️  No label for {img_path.name}")
        continue

    with open(lbl_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            cls, cx, cy, bw, bh = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])

            # Convert YOLO → pixel coords
            x1 = int((cx - bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((cy + bh / 2) * h)

            name, color = CLASSES.get(cls, (str(cls), (255, 255, 255)))

            # Draw box and label
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, name, (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Resize if image is too large to fit screen
    max_dim = 900
    scale = min(max_dim / w, max_dim / h, 1.0)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)))

    cv2.imshow(img_path.name, img)
    print(f"✅ Showing: {img_path.name} — press any key for next image")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

print("✅ Done visualizing!")
