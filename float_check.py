import cv2, os
from ultralytics import YOLO

root = r"data\split\train"
name = "13000000_test"
img_path = os.path.join(root, "images", name + ".jpg")
lbl_path = os.path.join(root, "labels", name + ".txt")

m = YOLO(r"runs\pcb_yolov8m_relu\weights\best.pt")
names = m.names
img = cv2.imread(img_path)
assert img is not None, img_path
h, w = img.shape[:2]

print("GROUND TRUTH:")
if os.path.exists(lbl_path):
    for line in open(lbl_path):
        c, cx, cy, bw, bh = line.split()
        c = int(c); cx, cy, bw, bh = [float(v) for v in (cx, cy, bw, bh)]
        print(f"  {names[c]:16s} [{(cx-bw/2)*w:.0f},{(cy-bh/2)*h:.0f},{(cx+bw/2)*w:.0f},{(cy+bh/2)*h:.0f}]")
else:
    print("  no label file at", lbl_path)

def pre(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    c = clahe.apply(blurred)
    _, bw = cv2.threshold(c, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    r = cv2.resize(bw, (640, 640), interpolation=cv2.INTER_LINEAR)
    return cv2.merge([r, r, r])

def show(tag, im):
    r = m.predict(im, imgsz=640, conf=0.25, iou=0.45, verbose=False)[0]
    print(tag)
    for b in r.boxes:
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        print(f"  {names[int(b.cls)]:16s} {float(b.conf):.3f} [{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}]")

show("FLOAT, raw image:", img)
show("FLOAT, binarized (same input as board):", pre(img))
