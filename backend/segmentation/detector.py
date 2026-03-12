"""
Jewelry Component Detector
==========================
Uses GroundingDINO for open-set text-prompted detection of jewelry parts,
then SAM2 for high-quality segmentation masks per detected component.

Memory strategy (8GB VRAM):
  1. Load GroundingDINO (~2GB) → detect bounding boxes → offload to CPU
  2. Load SAM2 base_plus (~3-4GB) → generate masks per box → offload to CPU
  3. torch.cuda.empty_cache() between stages
"""

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

# Default text prompt covering common jewelry components
# GroundingDINO uses '.' as separator between phrases
DEFAULT_PROMPT = "gemstone . prong . metal band . stone setting . bail . clasp"

# Simplified label mapping: map detected phrases to canonical semantic labels
LABEL_MAP = {
    "gemstone": "gemstone",
    "gem": "gemstone",
    "stone": "gemstone",
    "diamond": "gemstone",
    "ruby": "gemstone",
    "sapphire": "gemstone",
    "emerald": "gemstone",
    "prong": "prong",
    "claw": "prong",
    "metal band": "metal",
    "band": "metal",
    "metal": "metal",
    "ring": "metal",
    "chain": "metal",
    "stone setting": "setting",
    "setting": "setting",
    "bezel": "setting",
    "bail": "bail",
    "loop": "bail",
    "clasp": "clasp",
    "hook": "clasp",
    "closure": "clasp",
}


def _normalize_label(phrase: str) -> str:
    """Map a detected phrase to a canonical semantic label."""
    phrase_lower = phrase.strip().lower()
    # Try exact match first, then substring match
    if phrase_lower in LABEL_MAP:
        return LABEL_MAP[phrase_lower]
    for key, label in LABEL_MAP.items():
        if key in phrase_lower or phrase_lower in key:
            return label
    return "metal"  # Default fallback — most jewelry surface is metal


class JewelryDetector:
    """
    Detects and segments jewelry components from a single image.

    Pipeline:
      1. GroundingDINO: text-prompted open-set detection → bounding boxes
      2. SAM2: box-prompted segmentation → per-component binary masks

    All models are loaded lazily and offloaded to CPU after use to stay
    within 8GB VRAM budget.
    """

    def __init__(
        self,
        device: str = "cuda",
        grounding_dino_config: Optional[str] = None,
        grounding_dino_checkpoint: Optional[str] = None,
        sam2_checkpoint: Optional[str] = None,
        sam2_config: str = "configs/sam2.1/sam2.1_hiera_b+",
        box_threshold: float = 0.25,
        text_threshold: float = 0.20,
    ):
        self.device = device
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

        # Model paths — defaults point to models/ directory
        models_dir = Path(__file__).parent.parent.parent / "models"

        # GroundingDINO config: prefer pip package location, fall back to cloned repo
        if grounding_dino_config:
            self.gdino_config = grounding_dino_config
        else:
            try:
                import groundingdino as _gdino_pkg
                _pkg_config = Path(_gdino_pkg.__file__).parent / "config" / "GroundingDINO_SwinT_OGC.py"
                if _pkg_config.exists():
                    self.gdino_config = str(_pkg_config)
                else:
                    self.gdino_config = str(models_dir / "GroundingDINO" / "groundingdino" / "config" / "GroundingDINO_SwinT_OGC.py")
            except ImportError:
                self.gdino_config = str(models_dir / "GroundingDINO" / "groundingdino" / "config" / "GroundingDINO_SwinT_OGC.py")

        self.gdino_checkpoint = grounding_dino_checkpoint or str(
            models_dir / "groundingdino_swint_ogc.pth"
        )
        self.sam2_checkpoint = sam2_checkpoint or str(
            models_dir / "sam2.1_hiera_base_plus.pt"
        )
        self.sam2_config = sam2_config

        # Models are loaded lazily
        self._gdino_model = None
        self._sam2_predictor = None

    def _load_grounding_dino(self):
        """Load GroundingDINO model to GPU."""
        if self._gdino_model is not None:
            self._gdino_model.to(self.device)
            return

        logger.info("Loading GroundingDINO (SwinT) → %s", self.device)
        try:
            from groundingdino.util.inference import load_model
            self._gdino_model = load_model(
                self.gdino_config,
                self.gdino_checkpoint,
                device=self.device,
            )
        except ImportError:
            raise RuntimeError(
                "GroundingDINO not installed. Run:\n"
                "  git clone https://github.com/IDEA-Research/GroundingDINO.git\n"
                "  cd GroundingDINO && pip install -e ."
            )
        logger.info("GroundingDINO loaded (~2GB VRAM)")

    def _offload_grounding_dino(self):
        """Move GroundingDINO to CPU and free VRAM."""
        if self._gdino_model is not None:
            self._gdino_model.to("cpu")
            torch.cuda.empty_cache()
            logger.info("GroundingDINO offloaded to CPU")

    def _load_sam2(self):
        """Load SAM2 image predictor to GPU."""
        if self._sam2_predictor is not None:
            self._sam2_predictor.model.to(self.device)
            return

        logger.info("Loading SAM2 (base_plus) → %s", self.device)
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            sam2_model = build_sam2(
                self.sam2_config,
                self.sam2_checkpoint,
                device=self.device,
            )
            self._sam2_predictor = SAM2ImagePredictor(sam2_model)
        except ImportError:
            raise RuntimeError(
                "SAM2 not installed. Run:\n"
                "  git clone https://github.com/facebookresearch/sam2.git\n"
                "  cd sam2 && pip install -e ."
            )
        logger.info("SAM2 loaded (~3-4GB VRAM)")

    def _offload_sam2(self):
        """Move SAM2 to CPU and free VRAM."""
        if self._sam2_predictor is not None:
            self._sam2_predictor.model.to("cpu")
            torch.cuda.empty_cache()
            logger.info("SAM2 offloaded to CPU")

    def detect(
        self,
        image: Image.Image,
        prompt: str = DEFAULT_PROMPT,
    ) -> dict:
        """
        Detect and segment jewelry components in an image.

        Args:
            image: PIL Image (RGB)
            prompt: GroundingDINO text prompt with '.' separators

        Returns:
            {
                "parts": {
                    "gemstone": {
                        "mask": np.ndarray (H, W) bool,
                        "bbox": [x1, y1, x2, y2],  # absolute pixel coords
                        "confidence": float
                    },
                    "metal": { ... },
                    ...
                },
                "image_size": (H, W),
                "all_masks": np.ndarray (H, W) int  # combined label map (0=background)
            }
        """
        # Convert to RGB numpy array for processing
        image_rgb = image.convert("RGB")
        image_np = np.array(image_rgb)
        h, w = image_np.shape[:2]

        # ─── Stage 1: GroundingDINO Detection ───────────────────────────
        self._load_grounding_dino()

        boxes, confidences, phrases = self._run_grounding_dino(image_np, prompt)
        self._offload_grounding_dino()

        if len(boxes) == 0:
            logger.warning("No jewelry components detected — returning full-image 'metal' mask")
            return self._fallback_result(h, w)

        # Convert normalized boxes (cx, cy, w, h) → absolute (x1, y1, x2, y2)
        abs_boxes = self._convert_boxes(boxes, w, h)

        # ─── Stage 2: SAM2 Segmentation ─────────────────────────────────
        self._load_sam2()

        parts = {}
        label_map = np.zeros((h, w), dtype=np.int32)  # Combined label map
        label_index = 1  # 0 = background

        with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
            self._sam2_predictor.set_image(image_np)

            for i, (box, conf, phrase) in enumerate(zip(abs_boxes, confidences, phrases)):
                label = _normalize_label(phrase)

                # Run SAM2 with box prompt
                masks, scores, _ = self._sam2_predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=box[None, :],  # SAM2 expects [1, 4] shaped box
                    multimask_output=False,
                )

                # masks shape: (1, H, W) — take first (only) mask
                mask = masks[0].astype(bool)

                if label in parts:
                    # Merge masks for same semantic label (e.g., multiple prongs)
                    parts[label]["mask"] = parts[label]["mask"] | mask
                    # Keep the higher-confidence bbox
                    if conf > parts[label]["confidence"]:
                        parts[label]["bbox"] = box.tolist()
                        parts[label]["confidence"] = float(conf)
                else:
                    parts[label] = {
                        "mask": mask,
                        "bbox": box.tolist(),
                        "confidence": float(conf),
                    }

                # Update combined label map
                label_map[mask] = label_index
                label_index += 1

        self._offload_sam2()

        # If no "metal" detected, assign unlabeled pixels as metal
        # (most of a jewelry piece is metal)
        if "metal" not in parts:
            all_detected = np.zeros((h, w), dtype=bool)
            for part in parts.values():
                all_detected |= part["mask"]
            # Create a rough metal mask: everything not detected as another part
            # but within the jewelry bounding region
            metal_mask = ~all_detected & (label_map == 0)
            if metal_mask.any():
                parts["metal"] = {
                    "mask": metal_mask,
                    "bbox": [0, 0, w, h],
                    "confidence": 0.5,
                }

        return {
            "parts": parts,
            "image_size": (h, w),
            "all_masks": label_map,
        }

    def _run_grounding_dino(
        self, image_np: np.ndarray, prompt: str
    ) -> tuple:
        """Run GroundingDINO detection and return boxes, confidences, phrases."""
        from groundingdino.util.inference import predict as gdino_predict
        import groundingdino.datasets.transforms as T

        # GroundingDINO expects a specific transform
        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        # Convert numpy → PIL → transformed tensor
        image_pil = Image.fromarray(image_np)
        image_transformed, _ = transform(image_pil, None)

        boxes, logits, phrases = gdino_predict(
            model=self._gdino_model,
            image=image_transformed,
            caption=prompt,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            device=self.device,
        )

        return boxes.cpu().numpy(), logits.cpu().numpy(), phrases

    @staticmethod
    def _convert_boxes(
        boxes: np.ndarray, img_w: int, img_h: int
    ) -> np.ndarray:
        """
        Convert GroundingDINO boxes from normalized (cx, cy, w, h)
        to absolute pixel coords (x1, y1, x2, y2).
        """
        # boxes: (N, 4) in [cx, cy, w, h] normalized 0-1
        abs_boxes = np.zeros_like(boxes)
        abs_boxes[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2) * img_w  # x1
        abs_boxes[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2) * img_h  # y1
        abs_boxes[:, 2] = (boxes[:, 0] + boxes[:, 2] / 2) * img_w  # x2
        abs_boxes[:, 3] = (boxes[:, 1] + boxes[:, 3] / 2) * img_h  # y2
        return abs_boxes

    @staticmethod
    def _fallback_result(h: int, w: int) -> dict:
        """Return a fallback result when no components are detected."""
        return {
            "parts": {
                "metal": {
                    "mask": np.ones((h, w), dtype=bool),
                    "bbox": [0, 0, w, h],
                    "confidence": 1.0,
                }
            },
            "image_size": (h, w),
            "all_masks": np.ones((h, w), dtype=np.int32),
        }

    def cleanup(self):
        """Free all GPU memory."""
        self._offload_grounding_dino()
        self._offload_sam2()
        self._gdino_model = None
        self._sam2_predictor = None
        torch.cuda.empty_cache()
