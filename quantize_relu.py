import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pytorch_nndct.apis import torch_quantizer
from calib_dataset import PCBCalibDataset
from ultralytics.nn.tasks import DetectionModel  # ensures class is importable for unpickling

# --- Wrapper module ---
class QuantWrapper(nn.Module):
    def __init__(self, yolo_model):
        super().__init__()
        self.model = yolo_model

    def forward(self, x):
        out = self.model(x)
        boxes = out[1]['boxes']    # [1, 64, 8400] raw DFL logits
        scores = out[1]['scores']  # [1, 6, 8400]  raw class logits
        return boxes, scores

# --- Config ---
CKPT_PATH = "runs/pcb_yolov8m_relu/weights/best.pt"   # ReLU fine-tuned weights
OUT_DIR = "quantize_result_relu_raw"                        # new folder, old quantize_result stays untouched
CALIB_DIR = "data/calib_images"
IMG_SIZE = 640
BATCH_SIZE = 1  # Vitis AI quantizer calibration typically uses batch=1
QUANT_MODE = sys.argv[1] if len(sys.argv) > 1 else "test"  # run: python quantize_relu.py calib, then: python quantize_relu.py test
assert QUANT_MODE in ("calib", "test"), "Usage: python quantize_relu.py [calib|test]"

def load_model():
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    model = ckpt["model"].float()
    model.eval()
    n_silu = sum(isinstance(m, nn.SiLU) for m in model.modules())
    assert n_silu == 0, "SiLU found in model - use the ReLU fine-tuned best.pt"
    print("Checkpoint OK: no SiLU,", sum(isinstance(m, nn.ReLU) for m in model.modules()), "ReLU module(s)")
    return QuantWrapper(model)

def main():
    model = load_model()
    dummy_input = torch.randn(BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE)

    quantizer = torch_quantizer(
        quant_mode=QUANT_MODE,
        module=model,
        input_args=(dummy_input,),
        output_dir=OUT_DIR
    )
    quant_model = quantizer.quant_model

    dataset = PCBCalibDataset(CALIB_DIR, img_size=IMG_SIZE)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    if QUANT_MODE == "calib":
        with torch.no_grad():
            for i, imgs in enumerate(loader):
                _ = quant_model(imgs)
                if i % 50 == 0:
                    print(f"Calibration batch {i}/{len(loader)}")
        quantizer.export_quant_config()
        print("Calibration complete. Config exported.")

    else:
        # Test mode only needs ONE forward pass to finalize/export the xmodel
        with torch.no_grad():
            single_batch = next(iter(loader))
            _ = quant_model(single_batch)

        print(">>> About to call export_xmodel...")
        try:
            quantizer.export_xmodel(output_dir=OUT_DIR, deploy_check=True)
            print(">>> export_xmodel() returned without exception.")
        except Exception as e:
            print(f">>> EXPORT FAILED WITH EXCEPTION: {e}")
            import traceback
            traceback.print_exc()

        print(f">>> Listing {OUT_DIR}/ contents now:")
        import subprocess
        subprocess.run(["ls", "-la", OUT_DIR + "/"])

if __name__ == "__main__":
    main()
