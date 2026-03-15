import sys
from pathlib import Path
from typing import Dict, List

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from backend.reconstruction.pipeline import ReconstructionPipeline


SAMPLE_DIR = Path("samples")
OUTPUT_DIR = Path("temp/demo_validation")
CATEGORIES = {
    "ring": ["ring"],
    "pendant": ["pendant"],
    "necklace": ["necklace"],
}
EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def discover_images() -> Dict[str, List[Path]]:
    files = [p for p in SAMPLE_DIR.iterdir() if p.is_file() and p.suffix.lower() in EXTS]
    result: Dict[str, List[Path]] = {key: [] for key in CATEGORIES}

    for file_path in files:
        lower_name = file_path.name.lower()
        for category, keywords in CATEGORIES.items():
            if any(keyword in lower_name for keyword in keywords):
                result[category].append(file_path)

    return result


def run() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    discovered = discover_images()
    print("=== Demo Input Coverage ===")
    for category, files in discovered.items():
        if files:
            print(f"{category}: {len(files)} file(s) found")
        else:
            print(f"{category}: MISSING (add at least 1 sample image)")

    available = [path for files in discovered.values() for path in files]
    if not available:
        print("No supported sample images found in samples/.")
        return

    pipeline = ReconstructionPipeline(device="cuda")

    print("\n=== Preprocess Validation (rembg + TripoSR input) ===")
    for image_path in available:
        image = Image.open(image_path).convert("RGB")
        clean = pipeline._remove_background(image)
        triposr_input = pipeline._prepare_for_triposr(clean)

        target_dir = OUTPUT_DIR / image_path.stem
        target_dir.mkdir(parents=True, exist_ok=True)
        clean.save(target_dir / "input_clean.png")
        triposr_input.save(target_dir / "triposr_input_debug.png")

        print(f"{image_path.name}: saved {target_dir / 'input_clean.png'} and {target_dir / 'triposr_input_debug.png'}")


if __name__ == "__main__":
    run()
