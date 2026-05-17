import cv2
import glob
import random
import time
import os
from pathlib import Path
from ultralytics import YOLO

# ── Config ─────────────────────────────────────────────────────────────────
MODEL_PATH   = r"C:\Users\vagee\OneDrive\Desktop\major project\pcb-defect-detection\runs\pcb_yolov8m_v2\weights\best.pt"
TEST_IMAGES  = r"C:\Users\vagee\OneDrive\Desktop\major project\pcb-defect-detection\data\split\test\images"
PHONE_IP     = "http://10.238.202.94:8080//video"   # change to your IP Webcam IP
CONF         = 0.50
IMGSZ        = 640

CLASSES = ['missing_hole', 'mouse_bite', 'open_circuit', 'short', 'spur', 'spurious_copper']
COLORS  = {
    'missing_hole':    (255, 50,  50),
    'mouse_bite':      (50,  255, 50),
    'open_circuit':    (50,  50,  255),
    'short':           (255, 255, 50),
    'spur':            (50,  255, 255),
    'spurious_copper': (255, 50,  255),
}

# ── Load model ─────────────────────────────────────────────────────────────
print("=" * 60)
print("   PCB DEFECT DETECTION SYSTEM")
print("   YOLOv8m | 6 Defect Classes | mAP50: 99%")
print("=" * 60)
print("\n⏳ Loading model...")
model = YOLO(MODEL_PATH)
print("✅ Model loaded!\n")

# ── Menu ───────────────────────────────────────────────────────────────────
def show_menu():
    print("\n" + "─" * 40)
    print("  SELECT DEMO MODE:")
    print("  [1] Test Image Detection")
    print("  [2] Live Camera Detection (Phone)")
    print("  [3] Live Camera Detection (Webcam)")
    print("  [Q] Quit")
    print("─" * 40)
    return input("  Enter choice: ").strip().upper()

# ── Mode 1: Test Images ────────────────────────────────────────────────────
def demo_test_images():
    images = glob.glob(f"{TEST_IMAGES}/*.jpg")
    if not images:
        print("❌ No test images found. Check TEST_IMAGES path.")
        return

    random.shuffle(images)
    print(f"\n✅ Found {len(images)} test images")
    print("Press any key for next image | Q to go back to menu\n")

    for img_path in images:
        img = cv2.imread(img_path)
        if img is None:
            continue

        start = time.time()
        results = model(img, conf=CONF, imgsz=IMGSZ, verbose=False)
        elapsed = (time.time() - start) * 1000

        annotated = results[0].plot()

        # Stats overlay
        detections = results[0].boxes
        det_count = len(detections) if detections is not None else 0

        # Header bar
        cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 50), (20, 20, 20), -1)
        cv2.putText(annotated, f"PCB Defect Detection  |  Defects: {det_count}  |  {elapsed:.1f}ms",
                    (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        print(f"📋 {Path(img_path).name} → {det_count} defects detected ({elapsed:.1f}ms)")

        # Print each detection
        if detections is not None and len(detections):
            for box in detections:
                cls_id = int(box.cls[0])
                conf   = float(box.conf[0])
                print(f"   → {CLASSES[cls_id]}: {conf:.2f}")

        cv2.imshow("PCB Defect Detection — Test Images", annotated)
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q') or key == ord('Q'):
            break

    cv2.destroyAllWindows()

# ── Mode 2 & 3: Live Camera ────────────────────────────────────────────────
def demo_live(source):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"❌ Could not open camera source: {source}")
        print("   For phone: make sure IP Webcam app is running and IP is correct")
        return

    print("\n✅ Camera connected!")
    print("Controls: Q = quit | P = preprocess on/off | S = save screenshot\n")

    preprocess = True
    frame_count = 0
    fps_start = time.time()
    fps = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Lost camera feed")
            break

        # FPS calculation
        frame_count += 1
        if frame_count % 30 == 0:
            fps = 30 / (time.time() - fps_start)
            fps_start = time.time()

        # Preprocessing
        if preprocess:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Step 1 - Denoise first
            denoised = cv2.GaussianBlur(gray, (5, 5), 0)

            # Step 2 - CLAHE (contrast enhancement)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(denoised)

            # Step 3 - Otsu threshold (auto finds best threshold)
            _, binary = cv2.threshold(enhanced, 0, 255,
                                      cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            input_frame = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        else:
            input_frame = frame

        # Detection
        results = model(input_frame, conf=CONF, imgsz=IMGSZ, verbose=False)
        annotated = results[0].plot()

        detections = results[0].boxes
        det_count = len(detections) if detections is not None else 0

        # Overlay
        h, w = annotated.shape[:2]
        cv2.rectangle(annotated, (0, 0), (w, 55), (15, 15, 15), -1)
        cv2.putText(annotated, "PCB DEFECT DETECTION",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        cv2.putText(annotated,
                    f"FPS: {fps:.1f}  |  Defects: {det_count}  |  Preprocess: {'ON' if preprocess else 'OFF'}",
                    (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Alert if defects found
        if det_count > 0:
            cv2.rectangle(annotated, (w-220, 0), (w, 35), (0, 0, 180), -1)
            cv2.putText(annotated, f"⚠ {det_count} DEFECT(S) FOUND",
                        (w-215, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        cv2.imshow("PCB Defect Detection — Live", annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            break
        elif key == ord('p') or key == ord('P'):
            preprocess = not preprocess
            print(f"  Preprocessing: {'ON' if preprocess else 'OFF'}")
        elif key == ord('s') or key == ord('S'):
            fname = f"screenshot_{int(time.time())}.jpg"
            cv2.imwrite(fname, annotated)
            print(f"  📸 Saved: {fname}")

    cap.release()
    cv2.destroyAllWindows()

# ── Main Loop ──────────────────────────────────────────────────────────────
while True:
    choice = show_menu()

    if choice == '1':
        demo_test_images()
    elif choice == '2':
        print(f"\n📱 Connecting to phone camera: {PHONE_IP}")
        print("   Make sure IP Webcam app is running on your phone!")
        demo_live(PHONE_IP)
    elif choice == '3':
        print("\n💻 Connecting to laptop webcam...")
        demo_live(0)
    elif choice == 'Q':
        print("\n👋 Exiting. Good luck with your demo!\n")
        break
    else:
        print("❌ Invalid choice. Try again.")
