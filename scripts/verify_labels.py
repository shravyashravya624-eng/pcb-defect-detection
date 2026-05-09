import cv2
from pathlib import Path

labeled = Path(r"C:\Users\Admin\pcb-defect-detection\data\labeled")
imgs = list(labeled.glob("*_test.jpg"))[:3]

classes = ["missing_hole","mouse_bite","open_circuit","short","spur","spurious_copper"]
colors = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(0,255,255),(255,0,255)]

for img_path in imgs:
    txt_path = labeled / (img_path.stem + ".txt")
    if not txt_path.exists():
        continue
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    with open(txt_path) as f:
        for line in f:
            cls, cx, cy, bw, bh = map(float, line.strip().split())
            cls = int(cls)
            x1 = int((cx - bw/2) * w)
            y1 = int((cy - bh/2) * h)
            x2 = int((cx + bw/2) * w)
            y2 = int((cy + bh/2) * h)
            cv2.rectangle(img, (x1,y1), (x2,y2), colors[cls], 2)
            cv2.putText(img, classes[cls], (x1,y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[cls], 1)
    out = str(labeled.parent / f"check_{img_path.name}")
    cv2.imwrite(out, img)
    print(f"Saved: {out}")