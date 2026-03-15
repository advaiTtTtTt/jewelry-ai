import os
import sys
from pathlib import Path

import torch
import numpy as np
import trimesh
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from backend.reconstruction.pipeline import ReconstructionPipeline

def check_image(img_path):
    img = Image.open(img_path)
    arr = np.array(img)
    print(f"[{os.path.basename(img_path)}] Size: {img.size}, Mode: {img.mode}")
    if img.mode == 'RGBA':
        alpha = arr[:, :, 3]
        unique, counts = np.unique(alpha, return_counts=True)
        print("  Alpha distribution:", dict(zip(unique, counts)))
    elif img.mode == 'RGB':
        print("  Color bounds:", arr.min(axis=(0,1)), arr.max(axis=(0,1)))

pipeline = ReconstructionPipeline(device="cuda")

img = Image.open("samples/ring.png").convert("RGBA")

# 1. Background removal
print("--- STAGE A: Background Removal ---")
clean_img = pipeline._remove_background(img)
clean_img.save("diag_clean.png")
check_image("diag_clean.png")

# 2. TripoSR Prep
print("\n--- STAGE B: TripoSR Input ---")
image = clean_img
if image.mode == "RGBA":
    from tsr.utils import resize_foreground
    image = resize_foreground(image, 0.7)
    image = image.resize((512, 512), Image.Resampling.LANCZOS)
    bg = Image.new("RGB", image.size, (255, 255, 255))
    bg.paste(image, mask=image.split()[3])
    image = bg

image.save("diag_triposr_in.png")
check_image("diag_triposr_in.png")

# 3. TripoSR execution
print("\n--- STAGE C: TripoSR Recon ---")
pipeline._load_triposr()
pipeline._triposr_model.renderer.set_chunk_size(8192)

with torch.inference_mode():
    scene_codes = pipeline._triposr_model([image], device=pipeline.device)
    meshes = pipeline._triposr_model.extract_mesh(
        scene_codes,
        has_vertex_color=True,
        resolution=256,
        threshold=25.0,
    )

mesh = meshes[0]
mesh.export("diag_mesh.glb")
print("Mesh stats:", len(mesh.vertices), "vertices,", len(mesh.faces), "faces")
bounds = mesh.bounds
print("Mesh bounds:", bounds)
