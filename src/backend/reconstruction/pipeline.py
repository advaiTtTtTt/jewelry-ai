"""
3D Reconstruction Pipeline
===========================
Hybrid Zero123++ → TripoSR pipeline optimized for 8GB VRAM (RTX 4060 Laptop).

Pipeline stages (sequential, one model in VRAM at a time):
  Stage A: Background removal via rembg (~0.5GB)
  Stage B: Zero123++ multi-view generation (~5GB) → 6 views
  Stage C: TripoSR mesh reconstruction (~6GB) → GLB mesh
  Stage D: UV projection to map 2D segmentation masks → 3D vertex labels
  Stage E: GLB metadata injection via pygltflib

Each stage offloads its model to CPU before the next stage loads.
"""

import io
import json
import logging
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import trimesh
from PIL import Image

from .gem_builder import GemBuilder

PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Add TripoSR source to Python path (cloned at models/TripoSR)
_TRIPOSR_DIR = PROJECT_ROOT / "models" / "TripoSR"
if _TRIPOSR_DIR.exists() and str(_TRIPOSR_DIR) not in sys.path:
    sys.path.insert(0, str(_TRIPOSR_DIR))

logger = logging.getLogger(__name__)

# Zero123++ camera parameters for the 6 generated views
# Output is a 3×2 grid at 320×320 per view (640×960 total)
ZERO123_VIEWS = [
    {"azimuth": 30,  "elevation": 20,  "name": "front_right"},
    {"azimuth": 90,  "elevation": -10, "name": "right"},
    {"azimuth": 150, "elevation": 20,  "name": "back_right"},
    {"azimuth": 210, "elevation": -10, "name": "back_left"},
    {"azimuth": 270, "elevation": 20,  "name": "left"},
    {"azimuth": 330, "elevation": -10, "name": "front_left"},
]
ZERO123_FOV = 30  # degrees
VIEW_SIZE = 320   # pixels per view


class ReconstructionPipeline:
    """
    Full 2D→3D reconstruction pipeline for jewelry images.

    Usage:
        pipeline = ReconstructionPipeline(device="cuda")
        result = pipeline.reconstruct(image, segmentation_dict)
        # result = {"glb_path": str, "vertex_labels": dict, "views": [PIL.Image]}
    """

    def __init__(
        self,
        device: str = "cuda",
        output_dir: Optional[str] = None,
        triposr_chunk_size: int = 4096,  # Reduced from 8192 to save VRAM
    ):
        self.device = device
        self.output_dir = Path(output_dir or tempfile.mkdtemp(prefix="jewelry_"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.triposr_chunk_size = triposr_chunk_size

        # Lazy-loaded model references
        self._zero123_pipeline = None
        self._triposr_model = None

    # ═══════════════════════════════════════════════════════════════════
    # PUBLIC API
    # ═══════════════════════════════════════════════════════════════════

    def reconstruct(
        self,
        image: Image.Image,
        segmentation: Optional[dict] = None,
        quality: str = "high",
    ) -> dict:
        """
        Pure Procedural Pipeline: Uses Gemini VLM to extract a mathematical blueprint, 
        and generates flawless CAD-grade parametric meshes.
        """
        import json
        import tempfile
        import uuid

        job_id = str(uuid.uuid4())[:8]
        job_dir = self.output_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        logger.info("[%s] Starting Full Parametric CAD Pipeline", job_id)

        # Stage A: Background removal
        clean_image = self._remove_background(image)
        clean_image.save(job_dir / "input_clean.png")

        # Stage B: Extract structural CAD parameters via Gemini
        logger.info("[%s] Stage B: Vision LLM Parametric blueprint extraction", job_id)
        from src.backend.reconstruction.gemini_vision import analyze_jewelry_image
        blueprint = analyze_jewelry_image(clean_image)

        # Stage C: Procedural generation using mathematical parameters
        logger.info("[%s] Stage C: Parametric mesh construction", job_id)
        from src.backend.reconstruction.ring_builder import RingBuilder
        builder = RingBuilder()
        meshes = builder.build_ring(blueprint)

        # Stage D: Combine meshes
        combined = trimesh.util.concatenate(meshes)
        
        raw_glb_path = job_dir / "raw_mesh.glb"
        combined.export(str(raw_glb_path))
        
        # Build vertex labels mapped to the concatenated mesh
        vertex_labels = {}
        offset = 0
        for m in meshes:
            label = m.metadata.get("semantic_label", "metal")
            for j in range(len(m.vertices)):
                vertex_labels[offset + j] = label
            offset += len(m.vertices)

        # Stage E: Inject vertex labels + PBR setup into GLB
        logger.info("[%s] Stage E: GLB metadata injection", job_id)
        final_glb_path = job_dir / "jewelry.glb"
        self._inject_glb_metadata(raw_glb_path, final_glb_path, vertex_labels)

        # Compress and save sidecar
        compressed_labels = self._compress_vertex_labels(vertex_labels)
        labels_path = job_dir / "vertex_labels.json"
        
        with open(labels_path, "w") as f:
            json.dump(compressed_labels, f)

        logger.info("[%s] Pipeline complete → %s", job_id, final_glb_path)

        return {
            "glb_path": str(final_glb_path),
            "vertex_labels": compressed_labels,
            "views": [clean_image],
            "job_id": job_id,
        }

    # ═══════════════════════════════════════════════════════════════════
    # STAGE A: Background Removal
    # ═══════════════════════════════════════════════════════════════════

    @staticmethod
    def _remove_background(image: Image.Image) -> Image.Image:
        """Remove background using rembg. Returns RGBA image."""
        from rembg import remove, new_session
        import numpy as np
        
        # Use isnet-general-use which performs much better on complex shapes and holes
        session = new_session("isnet-general-use")
        
        print("Before rembg:", image.mode, image.size)
        # Use alpha matting to gracefully fade soft shadows instead of leaving harsh edges
        result = remove(
            image, 
            session=session,
            alpha_matting=True,
            alpha_matting_foreground_threshold=240,
            alpha_matting_background_threshold=10,
            alpha_matting_erode_size=10
        )
        print("After rembg:", result.mode, result.size)
        
        result = result.convert("RGBA")
        
        arr = np.array(result)
        
        # Fallback: if rembg accidentally removed the entire image (e.g. unrecognizable simple shape)
        if not np.any(arr[:, :, 3] > 0):
            import logging
            logging.getLogger(__name__).warning("rembg removed entire image, falling back to original")
            return image.convert("RGBA")
        
        return Image.fromarray(arr, mode="RGBA")

    # ═══════════════════════════════════════════════════════════════════
    # STAGE B: Multi-View Generation (Zero123++)
    # ═══════════════════════════════════════════════════════════════════

    def _load_zero123(self):
        """Load Zero123++ diffusion pipeline to GPU."""
        if self._zero123_pipeline is not None:
            self._zero123_pipeline.to(self.device)
            return

        logger.info("Loading Zero123++ v1.1 → %s (~5GB VRAM)", self.device)
        from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler

        self._zero123_pipeline = DiffusionPipeline.from_pretrained(
            "sudo-ai/zero123plus-v1.1",
            custom_pipeline="sudo-ai/zero123plus-pipeline",
            torch_dtype=torch.float16,
        )
        # Use recommended scheduler
        self._zero123_pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
            self._zero123_pipeline.scheduler.config,
            timestep_spacing="trailing",
        )
        self._zero123_pipeline.to(self.device)
        logger.info("Zero123++ loaded")

    def _offload_zero123(self):
        """Move Zero123++ to CPU and free VRAM."""
        if self._zero123_pipeline is not None:
            self._zero123_pipeline.to("cpu")
            torch.cuda.empty_cache()
            logger.info("Zero123++ offloaded to CPU")

    @staticmethod
    def _mask_to_band_only(image: Image.Image, segmentation: dict) -> Image.Image:
        """Clear out gemstone regions so TripoSR only sees the band."""
        import numpy as np
        img_array = np.array(image).copy()
        if not segmentation or "parts" not in segmentation:
            return image
            
        for part_name, part_data in segmentation["parts"].items():
            if part_name in ["gemstone", "setting", "prong", "stone"]:
                mask = part_data["mask"]
                if isinstance(mask, list):
                    mask = np.array(mask)
                # Handle RGBA vs RGB
                if img_array.shape[-1] == 4:
                    img_array[mask] = [0, 0, 0, 0]
                else:
                    img_array[mask] = [255, 255, 255]
        return Image.fromarray(img_array)

    def _generate_multiviews(self, image: Image.Image) -> list:
        """
        Generate 6 multi-view images from a single input image.

        Returns: list of 6 PIL Images (320×320 each)
        """
        self._load_zero123()

        # Zero123++ expects a square input image
        # Resize to 320×320 maintaining aspect ratio with padding
        input_image = self._prepare_input_image(image, target_size=320)

        with torch.inference_mode():
            result = self._zero123_pipeline(
                input_image,
                num_inference_steps=75,
            )

        # Result is a single image: 640×960 grid (3 rows × 2 cols)
        grid_image = result.images[0]

        # Split into 6 individual views
        views = self._split_view_grid(grid_image)

        self._offload_zero123()
        return views

    @staticmethod
    def _prepare_input_image(image: Image.Image, target_size: int = 320) -> Image.Image:
        """Resize and pad image to square for Zero123++ input."""
        image = image.convert("RGBA")
        # Find bounding box of non-transparent pixels
        bbox = image.getbbox()
        if bbox:
            image = image.crop(bbox)

        # Resize to fit within target_size with padding
        w, h = image.size
        scale = (target_size * 0.85) / max(w, h)  # 85% fill
        new_w, new_h = int(w * scale), int(h * scale)
        image = image.resize((new_w, new_h), Image.LANCZOS)

        # Center on white (or transparent) background
        result = Image.new("RGBA", (target_size, target_size), (255, 255, 255, 0))
        offset_x = (target_size - new_w) // 2
        offset_y = (target_size - new_h) // 2
        result.paste(image, (offset_x, offset_y), image)

        return result

    @staticmethod
    def _split_view_grid(grid_image: Image.Image) -> list:
        """
        Split Zero123++ output grid (640×960) into 6 individual 320×320 views.
        Grid layout: 3 rows × 2 columns
        """
        grid_w, grid_h = grid_image.size  # Expected: 640×960
        view_w = grid_w // 2   # 320
        view_h = grid_h // 3   # 320

        views = []
        for row in range(3):
            for col in range(2):
                x1 = col * view_w
                y1 = row * view_h
                view = grid_image.crop((x1, y1, x1 + view_w, y1 + view_h))
                views.append(view)
        return views

    # ═══════════════════════════════════════════════════════════════════
    # STAGE C: 3D Mesh Reconstruction (TripoSR)
    # ═══════════════════════════════════════════════════════════════════

    @staticmethod
    def _prepare_for_triposr(rgba_image: Image.Image) -> Image.Image:
        """
        Prepare input for TripoSR:
          1) Compute a tight foreground mask
          2) Crop by tight bounding box
          3) Composite onto pure white background after crop
          4) Add padding so foreground fills ~60%
          4) Resize to exact 512x512 RGB
        """
        image = rgba_image.convert("RGBA")
        rgba_arr = np.array(image)
        alpha_array = rgba_arr[:, :, 3]

        # Tight mask: prefer strong alpha, otherwise fall back to non-white RGB detection
        tight_mask = alpha_array > 200
        if not np.any(tight_mask):
            rgb_arr = np.array(rgba_image.convert("RGB"))
            tight_mask = np.any(rgb_arr < 245, axis=2)

        # Guard: if no foreground survives, feed pure white canvas
        if not np.any(tight_mask):
            return Image.new("RGB", (512, 512), (255, 255, 255))

        rows = np.any(tight_mask, axis=1)
        cols = np.any(tight_mask, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        bbox = (int(cmin), int(rmin), int(cmax) + 1, int(rmax) + 1)

        cropped_rgba = image.crop(bbox)

        # Composite over pure white AFTER cropping
        background = Image.new("RGBA", cropped_rgba.size, (255, 255, 255, 255))
        background.paste(cropped_rgba, mask=cropped_rgba.split()[3])
        rgb_image = background.convert("RGB")

        # Add padding to target a ~60% foreground ratio for rings/jewelry
        w, h = rgb_image.size
        size = max(w, h)
        ratio = 0.6
        pad = int((size / ratio - size) / 2)
        padded_size = size + 2 * pad

        padded = Image.new("RGB", (padded_size, padded_size), (255, 255, 255))
        offset_x = pad + (size - w) // 2
        offset_y = pad + (size - h) // 2
        padded.paste(rgb_image, (offset_x, offset_y))

        return padded.resize((512, 512), Image.Resampling.LANCZOS)

    def _load_triposr(self):
        """Load TripoSR model to GPU."""
        if self._triposr_model is not None:
            self._triposr_model.to(self.device)
            return

        logger.info("Loading TripoSR → %s (~6GB VRAM, chunk_size=%d)", 
                     self.device, self.triposr_chunk_size)
        try:
            from tsr.system import TSR

            self._triposr_model = TSR.from_pretrained(
                "stabilityai/TripoSR",
                config_name="config.yaml",
                weight_name="model.ckpt",
            )
            self._triposr_model.renderer.set_chunk_size(self.triposr_chunk_size)
            self._triposr_model.to(self.device)
        except ImportError:
            raise RuntimeError(
                "TripoSR not installed. Run:\n"
                "  pip install git+https://github.com/VAST-AI-Research/TripoSR.git"
            )
        logger.info("TripoSR loaded")

    def _offload_triposr(self):
        """Move TripoSR to CPU and free VRAM."""
        if self._triposr_model is not None:
            self._triposr_model.to("cpu")
            torch.cuda.empty_cache()
            logger.info("TripoSR offloaded to CPU")

    def _reconstruct_mesh(
        self, 
        images: list[Image.Image] | Image.Image,
        quality: str = "high"
    ) -> trimesh.Trimesh:
        """
        Reconstruct 3D mesh from one or more preprocessed images using TripoSR.

        Returns: trimesh.Trimesh with vertices, faces, and vertex colors.
        """
        self._load_triposr()
        
        # Increase chunk size to ensure complete marching cubes geometry evaluation
        self._triposr_model.renderer.set_chunk_size(8192)

        if isinstance(images, Image.Image):
            images_to_triposr = [images]
        else:
            images_to_triposr = list(images)

        normalized_images = []
        for image in images_to_triposr:
            if image.mode != "RGB":
                image = image.convert("RGB")
            if image.size != (512, 512):
                image = image.resize((512, 512), Image.Resampling.LANCZOS)
            normalized_images.append(image)

        if not normalized_images:
            normalized_images = [Image.new("RGB", (512, 512), (255, 255, 255))]

        logger.info(
            "TripoSR Input - Count: %d, Mode: %s, Size: %s, Corner Pixel: %s",
            len(normalized_images),
            normalized_images[0].mode,
            normalized_images[0].size,
            normalized_images[0].getpixel((0, 0)),
        )
        print(f"Feeding {len(normalized_images)} images to TripoSR")

        with torch.inference_mode():
            scene_codes = self._triposr_model(normalized_images, device=self.device)

            # TripoSR is single-view by architecture (Nv=1), so fuse per-view scene codes.
            if scene_codes.shape[0] > 1:
                scene_codes = scene_codes.mean(dim=0, keepdim=True)

            res_map = {"low": 128, "medium": 192, "high": 256}
            resolution = res_map.get(quality, 256)

            # Extract mesh using marching cubes
            meshes = self._triposr_model.extract_mesh(
                scene_codes,
                has_vertex_color=True,
                resolution=resolution,       # Marching cubes grid resolution (256 captures thin rings better)
                threshold=30.0,       # Increased to 30.0 to drop low-density webbing
            )

        self._offload_triposr()

        # TripoSR returns a list of meshes (one per input image)
        mesh = meshes[0]


        # Keep only the largest connected component
        components = mesh.split(only_watertight=False)
        if components:
            mesh = max(components, key=lambda m: len(m.vertices))

        # Smooth foil-like surfaces
        try:
            import trimesh.smoothing
            # First use Taubin to preserve volume and reduce staircasing
            trimesh.smoothing.filter_taubin(mesh, iterations=50)
            # Then use a light Laplacian pass to iron out remaining high-frequency jagged edges
            trimesh.smoothing.filter_laplacian(mesh, iterations=10)
        except Exception as e:
            logger.warning("Failed to apply mesh smoothing: %s", e)

        # Fill holes in the mesh
        trimesh.repair.fill_holes(mesh)
        trimesh.repair.fix_normals(mesh)

        # Clean up the mesh
        mesh = self._clean_mesh(mesh)

        return mesh

    @staticmethod
    def _clean_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """Post-process mesh: remove degenerate faces, fill holes, smooth."""
        # Remove degenerate faces (trimesh 4.x API)
        mask = mesh.nondegenerate_faces()
        mesh.update_faces(mask)

        # Remove duplicate faces
        mesh.update_faces(mesh.unique_faces())

        # Remove unreferenced vertices
        mesh.remove_unreferenced_vertices()

        # Fill small holes (common in marching cubes output)
        trimesh.repair.fill_holes(mesh)

        # Make mesh watertight if possible
        trimesh.repair.fix_winding(mesh)
        trimesh.repair.fix_normals(mesh)

        logger.info(
            "Mesh cleaned: %d vertices, %d faces, watertight=%s",
            len(mesh.vertices), len(mesh.faces), mesh.is_watertight,
        )
        return mesh

    # ═══════════════════════════════════════════════════════════════════
    # STAGE D: 2D Segmentation → 3D Vertex Labels
    # ═══════════════════════════════════════════════════════════════════

    def _map_segmentation_to_vertices(
        self,
        mesh: trimesh.Trimesh,
        segmentation: dict,
        views: list,
    ) -> dict:
        """
        Map 2D segmentation masks onto 3D mesh vertices using UV projection.

        Strategy:
          1. Project vertices from front/near-front cameras only (since
             segmentation masks come from the front-facing original photo,
             side/back views would map onto wrong mask regions)
          2. Weight front-facing view heavily (5×) for accurate labeling
          3. Majority-vote across weighted views for each vertex
          4. Fill unlabeled vertices via nearest-neighbor propagation

        Returns: {vertex_index: semantic_label}
        """
        from collections import Counter

        vertices = mesh.vertices  # (N, 3)
        n_verts = len(vertices)

        # Collect votes for each vertex across views
        # label_votes[i] = list of labels from different views
        label_votes = [[] for _ in range(n_verts)]

        parts = segmentation.get("parts", {})
        img_h, img_w = segmentation.get("image_size", (320, 320))

        # Only use front-facing views for projection.
        # The segmentation masks are from the original photo (roughly front-on),
        # so side/back view projections land on wrong mask regions and inject noise.
        PROJECTION_VIEWS = [
            # Primary front view — matches the original photo angle best
            {"azimuth": 0,   "elevation": 0,   "weight": 5},
            # Near-front views with moderate weight
            {"azimuth": 30,  "elevation": 20,  "weight": 2},   # front_right
            {"azimuth": 330, "elevation": -10, "weight": 2},   # front_left
            # Slight side views with low weight (still somewhat visible from front)
            {"azimuth": 60,  "elevation": 10,  "weight": 1},
            {"azimuth": 300, "elevation": 10,  "weight": 1},
        ]

        # Sort parts by area (smallest first) so small details (gemstone, prong) override base (metal)
        sorted_parts = sorted(
            parts.items(),
            key=lambda item: item[1]["mask"].sum()
        )

        for view_params in PROJECTION_VIEWS:
            weight = view_params["weight"]

            # Build projection matrix for this view
            proj_matrix = self._build_projection_matrix(
                azimuth=view_params["azimuth"],
                elevation=view_params["elevation"],
                fov=ZERO123_FOV,
                img_w=img_w,
                img_h=img_h,
            )

            # Project all vertices to 2D
            px_coords, visible = self._project_vertices(
                vertices, proj_matrix, img_w, img_h
            )

            # For each visible vertex, check which mask it falls into
            for vert_idx in range(n_verts):
                if not visible[vert_idx]:
                    continue

                px, py = int(px_coords[vert_idx, 0]), int(px_coords[vert_idx, 1])

                # Bounds check
                if 0 <= px < img_w and 0 <= py < img_h:
                    for label, part_data in sorted_parts:
                        mask = part_data["mask"]
                        if mask[py, px]:
                            # Add 'weight' copies so front views count more
                            label_votes[vert_idx].extend([label] * weight)
                            break  # First matching mask wins for this view

        # Majority vote for each vertex
        vertex_labels = {}
        for i in range(n_verts):
            if label_votes[i]:
                counts = Counter(label_votes[i])
                vertex_labels[i] = counts.most_common(1)[0][0]

        # Fill unlabeled vertices via nearest-neighbor propagation
        vertex_labels = self._propagate_labels(vertices, vertex_labels, n_verts)

        # Log label distribution
        dist = Counter(vertex_labels.values())
        logger.info("Vertex label distribution: %s", dict(dist))

        return vertex_labels

    @staticmethod
    def _build_projection_matrix(
        azimuth: float, elevation: float, fov: float,
        img_w: int, img_h: int, distance: float = 2.0,
    ) -> np.ndarray:
        """
        Build a 3×4 projection matrix for a camera at given azimuth/elevation.

        The camera looks at the origin from distance `distance`.
        """
        az_rad = np.radians(azimuth)
        el_rad = np.radians(elevation)

        # Camera position in spherical coordinates
        cam_x = distance * np.cos(el_rad) * np.sin(az_rad)
        cam_y = distance * np.sin(el_rad)
        cam_z = distance * np.cos(el_rad) * np.cos(az_rad)
        cam_pos = np.array([cam_x, cam_y, cam_z])

        # Look-at matrix (camera looks at origin)
        forward = -cam_pos / np.linalg.norm(cam_pos)
        world_up = np.array([0, 1, 0], dtype=np.float64)
        right = np.cross(forward, world_up)
        if np.linalg.norm(right) < 1e-6:
            world_up = np.array([0, 0, 1], dtype=np.float64)
            right = np.cross(forward, world_up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)

        # View matrix (world → camera)
        view = np.eye(4)
        view[:3, 0] = right
        view[:3, 1] = up
        view[:3, 2] = -forward
        view[:3, 3] = -np.array([right @ cam_pos, up @ cam_pos, -forward @ cam_pos])

        # Intrinsic matrix (perspective projection)
        fov_rad = np.radians(fov)
        focal = img_w / (2 * np.tan(fov_rad / 2))
        intrinsic = np.array([
            [focal, 0,     img_w / 2],
            [0,     focal, img_h / 2],
            [0,     0,     1],
        ])

        # Combined projection: 3×4 = intrinsic(3×3) @ view[:3, :](3×4)
        proj = intrinsic @ view[:3, :]
        return proj

    @staticmethod
    def _project_vertices(
        vertices: np.ndarray,
        proj_matrix: np.ndarray,
        img_w: int,
        img_h: int,
    ) -> tuple:
        """
        Project 3D vertices to 2D pixel coordinates.

        Returns:
            px_coords: (N, 2) pixel coordinates
            visible: (N,) boolean — True if vertex is in front of camera and in-bounds
        """
        n = len(vertices)
        # Homogeneous coordinates (N, 4)
        verts_h = np.hstack([vertices, np.ones((n, 1))])

        # Project: (3, 4) @ (4, N) → (3, N)
        projected = (proj_matrix @ verts_h.T).T  # (N, 3)

        # Perspective divide
        z = projected[:, 2]
        visible = z > 0.01  # Must be in front of camera

        px_coords = np.zeros((n, 2))
        px_coords[visible, 0] = projected[visible, 0] / z[visible]  # x
        px_coords[visible, 1] = projected[visible, 1] / z[visible]  # y

        # Check bounds
        visible &= (px_coords[:, 0] >= 0) & (px_coords[:, 0] < img_w)
        visible &= (px_coords[:, 1] >= 0) & (px_coords[:, 1] < img_h)

        return px_coords, visible

    @staticmethod
    def _propagate_labels(
        vertices: np.ndarray,
        vertex_labels: dict,
        n_verts: int,
    ) -> dict:
        """
        Fill unlabeled vertices using nearest-neighbor propagation
        from already-labeled vertices.
        """
        if len(vertex_labels) == n_verts:
            return vertex_labels

        labeled_indices = list(vertex_labels.keys())
        if not labeled_indices:
            # No labels at all — everything is metal
            return {i: "metal" for i in range(n_verts)}

        labeled_positions = vertices[labeled_indices]
        labeled_labels = [vertex_labels[i] for i in labeled_indices]

        from scipy.spatial import KDTree
        tree = KDTree(labeled_positions)

        for i in range(n_verts):
            if i not in vertex_labels:
                # Find nearest labeled vertex
                _, idx = tree.query(vertices[i])
                vertex_labels[i] = labeled_labels[idx]

        return vertex_labels

    # ═══════════════════════════════════════════════════════════════════
    # STAGE E: GLB Metadata Injection
    # ═══════════════════════════════════════════════════════════════════

    @staticmethod
    def _inject_glb_metadata(
        input_path: Path,
        output_path: Path,
        vertex_labels: dict,
    ):
        """
        Inject semantic vertex labels and component groups into GLB file
        using pygltflib. This metadata is readable by Three.js GLTFLoader
        via gltf.scene.userData.

        The metadata structure:
        {
            "jewelry_ai": true,
            "vertex_labels": {"0": "metal", "1": "gemstone", ...},
            "semantic_groups": {
                "metal": [0, 1, 5, 6, ...],       # vertex indices
                "gemstone": [2, 3, 4, ...],
                "prong": [7, 8, ...],
                ...
            }
        }
        """
        import pygltflib

        gltf = pygltflib.GLTF2().load(str(input_path))

        # Build semantic groups from vertex labels
        semantic_groups = {}
        for vert_idx, label in vertex_labels.items():
            idx_key = str(vert_idx) if isinstance(vert_idx, int) else vert_idx
            if label not in semantic_groups:
                semantic_groups[label] = []
            semantic_groups[label].append(int(idx_key))

        # Compress groups into ranges for smaller JSON
        compressed_groups = {}
        for label, indices in semantic_groups.items():
            compressed_groups[label] = _compress_index_ranges(sorted(indices))

        # Store in scene extras (accessible via gltf.scene.userData in Three.js)
        metadata = {
            "jewelry_ai": True,
            "version": "1.0",
            "semantic_groups": compressed_groups,
            # Full vertex labels stored as range-based mapping
            "label_ranges": _build_label_ranges(vertex_labels),
        }

        if gltf.scenes and len(gltf.scenes) > 0:
            if gltf.scenes[0].extras is None:
                gltf.scenes[0].extras = {}
            gltf.scenes[0].extras.update(metadata)

        # Also store on each mesh node for per-mesh access
        for node_idx, node in enumerate(gltf.nodes):
            if node.mesh is not None:
                if node.extras is None:
                    node.extras = {}
                node.extras["jewelry_ai"] = True
                node.extras["semantic_groups"] = compressed_groups

        gltf.save(str(output_path))
        logger.info("GLB metadata injected → %s", output_path)

    @staticmethod
    def _compress_vertex_labels(vertex_labels: dict) -> dict:
        """
        Compress vertex labels for JSON serialization.
        Instead of {0: "metal", 1: "metal", 2: "metal", 3: "gemstone", ...}
        Output: {"metal": [[0, 2]], "gemstone": [[3, 3]], ...}  (inclusive ranges)
        """
        groups = {}
        for idx, label in sorted(vertex_labels.items(), key=lambda x: int(x[0])):
            idx = int(idx)
            if label not in groups:
                groups[label] = []
            groups[label].append(idx)

        compressed = {}
        for label, indices in groups.items():
            compressed[label] = _compress_index_ranges(indices)
        return compressed

    def cleanup(self):
        """Free all GPU memory."""
        self._offload_zero123()
        self._offload_triposr()
        self._zero123_pipeline = None
        self._triposr_model = None
        torch.cuda.empty_cache()


# ═══════════════════════════════════════════════════════════════════════
# Utility Functions
# ═══════════════════════════════════════════════════════════════════════

def _compress_index_ranges(sorted_indices: list) -> list:
    """
    Compress a sorted list of integers into ranges.
    [0, 1, 2, 5, 6, 10] → [[0, 2], [5, 6], [10, 10]]
    """
    if not sorted_indices:
        return []

    ranges = []
    start = sorted_indices[0]
    end = start

    for idx in sorted_indices[1:]:
        if idx == end + 1:
            end = idx
        else:
            ranges.append([start, end])
            start = idx
            end = idx
    ranges.append([start, end])
    return ranges


def _build_label_ranges(vertex_labels: dict) -> dict:
    """
    Build a compact range-based representation of vertex labels.
    Groups consecutive vertices with the same label into ranges.
    """
    if not vertex_labels:
        return {}

    # Sort by vertex index
    sorted_items = sorted(vertex_labels.items(), key=lambda x: int(x[0]))

    ranges = []
    current_label = sorted_items[0][1]
    range_start = int(sorted_items[0][0])
    range_end = range_start

    for idx_str, label in sorted_items[1:]:
        idx = int(idx_str)
        if label == current_label and idx == range_end + 1:
            range_end = idx
        else:
            ranges.append({"start": range_start, "end": range_end, "label": current_label})
            current_label = label
            range_start = idx
            range_end = idx

    ranges.append({"start": range_start, "end": range_end, "label": current_label})
    return ranges
