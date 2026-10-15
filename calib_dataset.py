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
        # Raw image, matching how the model was trained (RGB, 0-1, no binarization)
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.img_size, self.img_size),
                             interpolation=cv2.INTER_LINEAR)
        tensor = torch.from_numpy(resized).float() / 255.0
        return tensor.permute(2, 0, 1)  # CHW

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        img_bgr = cv2.imread(img_path)
        return self.preprocess(img_bgr)