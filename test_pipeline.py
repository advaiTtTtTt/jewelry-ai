import torch
import numpy as np
from PIL import Image
from backend.reconstruction.pipeline import ReconstructionPipeline

from backend.segmentation.detector import JewelryDetector

def run():
    img = Image.open('samples/ring.png').convert('RGB')
    detector = JewelryDetector('cuda')
    seg = detector.detect(img)
    pipeline = ReconstructionPipeline(device='cuda')
    result = pipeline.reconstruct(img, seg)
    print("DONE", result['glb_path'])

run()
