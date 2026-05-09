import cv2, random
from pathlib import Path

IMG_DIR = Path("data/labeled/train/images")
LBL_DIR = Path("data/labeled/train/labels")

CLASSES = ['Missing_hole', 'Mouse_bite', 'Open_circuit', 
           'Short', 'Spur', 'Spurious_copper']

COLORS = [(255,0,0),(0,255,0),(0,0,255),
          (255,255,0),(0,255,255),(255,0,255)]

images = list(IMG_DIR.glob("*.jpg"))
random.shuffle(images)  # random images every time

print(f"Total images: {len(images)}")
print("Press any key for next image, press Q to quit")

for img_path in images:
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    lbl_path = LBL_DIR / (img_path.stem + ".txt")
    
    if lbl_path.exists():
        with open(lbl_path) as f:
            for line in f:
                cls, cx, cy, bw, bh = map(float, line.strip().split())
                cls = int(cls)
                x1 = int((cx - bw/2) * w)
                y1 = int((cy - bh/2) * h)
                x2 = int((cx + bw/2) * w)
                y2 = int((cy + bh/2) * h)
                cv2.rectangle(img, (x1,y1), (x2,y2), COLORS[cls], 2)
                cv2.putText(img, CLASSES[cls], (x1, y1-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[cls], 1)
    
    cv2.imshow(img_path.name, img)
    key = cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    if key == ord('q') or key == ord('Q'):
        print("Quit!")
        break