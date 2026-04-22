import os
import cv2
from pathlib import Path

# DeepPCB class mapping (1-6 → 0-5 for YOLO)
CLASS_MAP = {
    1: 0,  # missing_hole
    2: 1,  # mouse_bite
    3: 2,  # open_circuit
    4: 3,  # short
    5: 4,  # spur
    6: 5   # spurious_copper
}

pcb_data = Path(r"C:\Users\Admin\pcb-defect-detection\data\raw\DeepPCB\PCBData")
output_dir = Path(r"C:\Users\Admin\pcb-defect-detection\data\labeled")
output_dir.mkdir(exist_ok=True)

converted = 0
skipped = 0

for group_folder in pcb_data.iterdir():
    if not group_folder.is_dir():
        continue
    
    not_folder = None
    img_folder = None
    
    for sub in group_folder.iterdir():
        if sub.name.endswith("_not"):
            not_folder = sub
        elif sub.is_dir():
            img_folder = sub
    
    if not not_folder or not img_folder:
        continue

    for txt_file in not_folder.glob("*.txt"):
        img_name = txt_file.stem  # e.g. 92000000
        img_path = img_folder / f"{img_name}_test.jpg"
        
        if not img_path.exists():
            skipped += 1
            continue
        
        # Get image dimensions
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
                yolo_cls = CLASS_MAP.get(cls, cls - 1)
                
                # Convert to YOLO format (normalized center x, y, w, h)
                cx = ((x1 + x2) / 2) / w
                cy = ((y1 + y2) / 2) / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                
                yolo_lines.append(f"{yolo_cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        
        if yolo_lines:
            # Save image
            out_img = output_dir / img_path.name
            import shutil
            shutil.copy(str(img_path), str(out_img))
            
            # Save YOLO label
            out_txt = output_dir / f"{img_name}_test.txt"
            with open(out_txt, "w") as f:
                f.write("\n".join(yolo_lines))
            converted += 1

print(f"✅ Done! Converted: {converted} images, Skipped: {skipped}")