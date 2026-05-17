import os
import shutil
import cv2
from pathlib import Path

# CORRECTED DeepPCB class mapping (from official README)
# 1=open, 2=short, 3=mouse_bite, 4=spur, 5=copper, 6=pin-hole
CLASS_MAP = {
    1: 2,  # open       → open_circuit
    2: 3,  # short      → short
    3: 1,  # mouse_bite  → mouse_bite
    4: 4,  # spur       → spur
    5: 5,  # copper     → spurious_copper
    6: 0,  # pin-hole   → missing_hole
}

pcb_data  = Path(r"C:\Users\vagee\OneDrive\Desktop\major project\pcb-defect-detection\data\raw\DeepPCB\PCBData")
output_dir = Path(r"C:\Users\vagee\OneDrive\Desktop\major project\pcb-defect-detection\data\labeled")

# Clear old labeled data
if output_dir.exists():
    shutil.rmtree(output_dir)
output_dir.mkdir(parents=True)

converted = skipped = 0

for group_folder in pcb_data.iterdir():
    if not group_folder.is_dir():
        continue
    not_folder = img_folder = None
    for sub in group_folder.iterdir():
        if sub.name.endswith("_not"):
            not_folder = sub
        elif sub.is_dir():
            img_folder = sub
    if not not_folder or not img_folder:
        continue

    for txt_file in not_folder.glob("*.txt"):
        img_path = img_folder / f"{txt_file.stem}_test.jpg"
        if not img_path.exists():
            skipped += 1
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            skipped += 1
            continue
        h, w = img.shape[:2]

        yolo_lines = []
        with open(txt_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                x1, y1, x2, y2, cls = map(int, parts)
                yolo_cls = CLASS_MAP.get(cls, -1)
                if yolo_cls == -1:
                    continue
                cx = ((x1 + x2) / 2) / w
                cy = ((y1 + y2) / 2) / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                yolo_lines.append(f"{yolo_cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        if yolo_lines:
            shutil.copy(str(img_path), output_dir / img_path.name)
            with open(output_dir / f"{txt_file.stem}_test.txt", "w") as f:
                f.write("\n".join(yolo_lines))
            converted += 1

print(f"✅ Done! Converted: {converted}, Skipped: {skipped}")