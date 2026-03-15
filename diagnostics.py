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

def run_diagnostics(image_path="samples/ring.png", out_dir="diagnostic_output"):
    os.makedirs(out_dir, exist_ok=True)
    out_dir = Path(out_dir)
    print(f"Loading {image_path}...")
    
    img = Image.open(image_path).convert("RGBA")
    
    pipeline = ReconstructionPipeline(device="cuda")
    
    # 1. Background Removal
    print("Running background removal...")
    clean_img = pipeline._remove_background(img)
    clean_img.save(out_dir / "1_background_removal.png")
    
    # 2. Multi-view generation
    print("Generating multi-views...")
    views = pipeline._generate_multiviews(clean_img)
    for i, view in enumerate(views):
        view.save(out_dir / f"2_multiview_{i}.png")
        
    # 3. TripoSR Prep
    print("Preparing image for TripoSR...")
    prep_img = clean_img
    if prep_img.mode == "RGBA":
        from tsr.utils import resize_foreground
        prep_img = resize_foreground(prep_img, 0.7)
        prep_img = prep_img.resize((512, 512), Image.Resampling.LANCZOS)
        bg = Image.new("RGB", prep_img.size, (255, 255, 255))
        bg.paste(prep_img, mask=prep_img.split()[3])
        prep_img = bg
    prep_img.save(out_dir / "3_triposr_input.png")
        
    # 4. Mesh Generation
    print("Running 3D mesh reconstruction...")
    mesh = pipeline._reconstruct_mesh(clean_img)
    mesh.export(str(out_dir / "4_raw_mesh.glb"))
    
    print("\n--- Diagnostic Results Saved to 'diagnostic_output' ---")
    print(f"Mesh Vertices: {len(mesh.vertices)}")
    print(f"Mesh Faces: {len(mesh.faces)}")
    print(f"Is Watertight (closed): {mesh.is_watertight}")
    print(f"Euler Number: {mesh.euler_number} (0 = Torus/Ring, 2 = Solid Sphere)")
    print("Inspect the images in 'diagnostic_output' to see where the process decays.")

if __name__ == "__main__":
    run_diagnostics()
