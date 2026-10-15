import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

class PCBCalibDataset(Dataset):
    def __init__(self, img_dir, img_size=640):
        self.img_dir = img_dir
        self.img_size = img_size
        self.img_files = [f for f in os.listdir(img_dir)
                           if f.lower().endswith((".jpg", ".png", ".bmp"))]

    def __len__(self):
        return len(self.img_files)

    def preprocess(self, img_bgr):
        # 1. Grayscale
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # 2. Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # 3. CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_img = clahe.apply(blurred)

        # 4. Otsu threshold (inverted, per your DeepPCB color-scheme fix)
        _, bw = cv2.threshold(clahe_img, 0, 255,
                               cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # 5. Resize to model input size
        resized = cv2.resize(bw, (self.img_size, self.img_size),
                              interpolation=cv2.INTER_LINEAR)

        # 6. Replicate single channel -> 3 channels
        rgb_like = cv2.merge([resized, resized, resized])

        # 7. Normalize to [0,1], HWC -> CHW
        tensor = torch.from_numpy(rgb_like).float() / 255.0
        tensor = tensor.permute(2, 0, 1)  # CHW
        return tensor

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        img_bgr = cv2.imread(img_path)
        return self.preprocess(img_bgr)