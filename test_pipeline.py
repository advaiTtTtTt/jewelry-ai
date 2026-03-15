import sys
from pathlib import Path

import torch
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from backend.reconstruction.pipeline import ReconstructionPipeline
from backend.segmentation.detector import JewelryDetector

def run():
    img = Image.open('samples/test.png').convert('RGB')
    detector = JewelryDetector('cuda')
    seg = detector.detect(img)
    pipeline = ReconstructionPipeline(device='cuda')
    result = pipeline.reconstruct(img, seg)
    print("DONE", result['glb_path'])

run()
