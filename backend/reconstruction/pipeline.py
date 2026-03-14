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

# Add TripoSR source to Python path (cloned at models/TripoSR)
_TRIPOSR_DIR = Path(__file__).resolve().parent.parent.parent / "models" / "TripoSR"
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
    ) -> dict:
        """
        Full reconstruction pipeline: image → GLB with semantic vertex labels.

        Args:
            image: Input PIL Image (RGB or RGBA)
            segmentation: Output from JewelryDetector.detect() (optional)

        Returns:
            {
                "glb_path": str,           # Path to output GLB file
                "vertex_labels": dict,     # {vertex_index: semantic_label}
                "views": [PIL.Image],      # 6 multi-view images
                "job_id": str,             # Unique identifier for this job
            }
        """
        job_id = str(uuid.uuid4())[:8]
        job_dir = self.output_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        logger.info("[%s] Starting reconstruction pipeline", job_id)

        # Stage A: Remove background
        logger.info("[%s] Stage A: Background removal", job_id)
        clean_image = self._remove_background(image)
        clean_image.save(job_dir / "input_clean.png")

        # Stage B: Generate multi-view images via Zero123++
        logger.info("[%s] Stage B: Multi-view generation (Zero123++)", job_id)
        views = self._generate_multiviews(clean_image)
        for i, view in enumerate(views):
            view.save(job_dir / f"view_{i}_{ZERO123_VIEWS[i]['name']}.png")

        # Stage C: Reconstruct 3D mesh via TripoSR
        logger.info("[%s] Stage C: 3D mesh reconstruction (TripoSR)", job_id)
        mesh = self._reconstruct_mesh(clean_image)
        raw_glb_path = job_dir / "raw_mesh.glb"
        mesh.export(str(raw_glb_path))

        # Stage D: Map 2D segmentation masks to 3D vertex labels
        logger.info("[%s] Stage D: Segmentation → vertex label mapping", job_id)
        if segmentation and "parts" in segmentation:
            vertex_labels = self._map_segmentation_to_vertices(
                mesh, segmentation, views
            )
        else:
            # No segmentation available — label everything as "metal"
            vertex_labels = {i: "metal" for i in range(len(mesh.vertices))}

        # Stage E: Inject vertex labels + PBR setup into GLB
        logger.info("[%s] Stage E: GLB metadata injection", job_id)
        final_glb_path = job_dir / "jewelry.glb"
        self._inject_glb_metadata(raw_glb_path, final_glb_path, vertex_labels)

        # Save vertex labels as JSON sidecar
        labels_path = job_dir / "vertex_labels.json"
        # Compress: group consecutive vertices with same label
        compressed_labels = self._compress_vertex_labels(vertex_labels)
        with open(labels_path, "w") as f:
            json.dump(compressed_labels, f)

        logger.info("[%s] Pipeline complete → %s", job_id, final_glb_path)

        return {
            "glb_path": str(final_glb_path),
            "vertex_labels": compressed_labels,
            "views": views,
            "job_id": job_id,
        }

    # ═══════════════════════════════════════════════════════════════════
    # STAGE A: Background Removal
    # ═══════════════════════════════════════════════════════════════════

    @staticmethod
    def _remove_background(image: Image.Image) -> Image.Image:
        """Remove background using rembg. Returns RGBA image."""
        from rembg import remove
        # rembg auto-downloads u2net model on first run (~176MB)
        result = remove(image)
        return result.convert("RGBA")

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

    def _reconstruct_mesh(self, image: Image.Image) -> trimesh.Trimesh:
        """
        Reconstruct 3D mesh from a single image using TripoSR.

        Returns: trimesh.Trimesh with vertices, faces, and vertex colors.
        """
        self._load_triposr()

        # TripoSR expects RGB (3-channel). If RGBA, composite on gray background.
        if image.mode == "RGBA":
            bg = Image.new("RGB", image.size, (127, 127, 127))
            bg.paste(image, mask=image.split()[3])  # Use alpha as mask
            image = bg
        elif image.mode != "RGB":
            image = image.convert("RGB")

        with torch.inference_mode():
            scene_codes = self._triposr_model([image], device=self.device)

            # Extract mesh using marching cubes (much lighter than FlexiCubes)
            meshes = self._triposr_model.extract_mesh(
                scene_codes,
                has_vertex_color=True,
                resolution=256,       # Marching cubes grid resolution
                threshold=25.0,       # Isosurface threshold
            )

        self._offload_triposr()

        # TripoSR returns a list of meshes (one per input image)
        mesh = meshes[0]

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
                    for label, part_data in parts.items():
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
