"""
Material Applier
================
Applies PBR materials to specific semantic components of a GLB mesh.

Key design: material swaps are done purely via glTF material editing —
NO AI re-calls, NO mesh regeneration. This makes swaps < 50ms.

Uses pygltflib for full glTF 2.0 extension support (transmission, IOR, volume)
that trimesh alone cannot handle.
"""

import io
import json
import logging
import struct
import time
from pathlib import Path
from typing import Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


class MaterialApplier:
    """
    Applies PBR materials to GLB meshes based on semantic vertex labels.

    The typical workflow:
      1. Load a GLB file that has semantic_groups in its extras metadata
      2. Split the mesh into per-component sub-meshes (one per semantic label)
      3. Apply a new material to a specific component
      4. Return the updated GLB as bytes

    All operations are in-memory and take < 50ms.
    """

    def apply_material(
        self,
        glb_input: Union[str, bytes],
        target_component: str,
        material_name: str,
        vertex_labels: Optional[dict] = None,
    ) -> bytes:
        """
        Apply a new material to a specific component in the GLB.

        Args:
            glb_input: Path to GLB file, or GLB bytes
            target_component: Semantic label to update ("metal", "gemstone", etc.)
            material_name: Material key from definitions.py ("yellow_gold", "diamond", etc.)
            vertex_labels: Optional pre-loaded vertex labels. If None, reads from GLB extras.

        Returns:
            Updated GLB file as bytes (in-memory, no disk I/O needed)
        """
        start = time.perf_counter()

        import pygltflib
        from ..materials.definitions import build_gltf_material_dict

        # Load GLB
        if isinstance(glb_input, (str, Path)):
            gltf = pygltflib.GLTF2().load(str(glb_input))
        else:
            gltf = pygltflib.GLTF2.load_from_bytes(glb_input)

        # Build the new material properties
        mat_dict = build_gltf_material_dict(material_name)

        # Find or create the material for target component
        mat_index = self._find_or_create_material(gltf, target_component)

        # Apply PBR properties to the material
        material = gltf.materials[mat_index]
        self._apply_pbr_properties(material, mat_dict)

        # Register required extensions in the glTF root
        self._register_extensions(gltf, mat_dict)

        # Assign this material to all primitives belonging to target_component
        self._assign_material_to_component(gltf, mat_index, target_component)

        # Serialize back to bytes
        glb_bytes = self._gltf_to_bytes(gltf)

        elapsed = (time.perf_counter() - start) * 1000
        logger.info(
            "Material '%s' applied to '%s' in %.1fms (%d bytes)",
            material_name, target_component, elapsed, len(glb_bytes),
        )

        return glb_bytes

    def split_mesh_by_labels(
        self,
        glb_input: Union[str, bytes],
        vertex_labels: dict,
    ) -> bytes:
        """
        Split a single-mesh GLB into multiple meshes based on semantic vertex labels.

        This is a one-time operation needed before per-component material swaps.
        After splitting, each semantic component has its own primitive with an
        independent material assignment.

        Args:
            glb_input: Path to GLB file, or GLB bytes
            vertex_labels: Compressed vertex labels dict
                           {"metal": [[0, 100], [200, 300]], "gemstone": [[101, 199]]}

        Returns:
            Updated GLB with split primitives as bytes
        """
        import trimesh
        import pygltflib

        # Load mesh with trimesh for geometry operations
        if isinstance(glb_input, (str, Path)):
            scene = trimesh.load(str(glb_input))
        else:
            scene = trimesh.load(io.BytesIO(glb_input), file_type="glb")

        # Get the mesh (might be a Scene with one geometry)
        if isinstance(scene, trimesh.Scene):
            mesh = trimesh.util.concatenate(tuple(scene.geometry.values()))
        else:
            mesh = scene

        # Expand ranges to full vertex-to-label mapping
        vert_to_label = self._expand_label_ranges(vertex_labels, len(mesh.vertices))

        # Create a face-to-label mapping (face takes the label of its majority vertices)
        face_labels = []
        for face in mesh.faces:
            face_vert_labels = [vert_to_label.get(v, "metal") for v in face]
            # Majority vote
            from collections import Counter
            label = Counter(face_vert_labels).most_common(1)[0][0]
            face_labels.append(label)

        # Split mesh by label
        unique_labels = sorted(set(face_labels))
        sub_meshes = {}
        for label in unique_labels:
            face_mask = np.array([l == label for l in face_labels])
            sub_mesh = mesh.submesh([face_mask], append=True)
            sub_meshes[label] = sub_mesh

        # Build a new scene with labeled sub-meshes
        new_scene = trimesh.Scene()
        for label, sub_mesh in sub_meshes.items():
            sub_mesh.metadata["semantic_label"] = label
            new_scene.add_geometry(sub_mesh, node_name=f"component_{label}")

        # Export as GLB
        glb_bytes = new_scene.export(file_type="glb")

        # Post-process with pygltflib to add semantic metadata + extensions
        gltf = pygltflib.GLTF2.load_from_bytes(glb_bytes)

        # Tag each node with its semantic label
        for node in gltf.nodes:
            if node.name and node.name.startswith("component_"):
                label = node.name.replace("component_", "")
                node.extras = {"semantic_label": label}

                # Create a named material for each component
                if node.mesh is not None:
                    mesh_obj = gltf.meshes[node.mesh]
                    mat_idx = self._find_or_create_material(gltf, label)
                    for prim in mesh_obj.primitives:
                        prim.material = mat_idx

        # Store metadata in scene extras
        if gltf.scenes and len(gltf.scenes) > 0:
            gltf.scenes[0].extras = {
                "jewelry_ai": True,
                "version": "1.0",
                "semantic_groups": {label: True for label in unique_labels},
                "label_ranges": vertex_labels if isinstance(vertex_labels, list) else None,
            }

        return self._gltf_to_bytes(gltf)

    # ═══════════════════════════════════════════════════════════════════
    # INTERNAL HELPERS
    # ═══════════════════════════════════════════════════════════════════

    @staticmethod
    def _find_or_create_material(gltf, component_name: str) -> int:
        """Find existing material for component, or create a new one."""
        import pygltflib

        # Search existing materials by name
        for i, mat in enumerate(gltf.materials):
            if mat.name and component_name in mat.name.lower():
                return i

        # Create new material
        new_mat = pygltflib.Material()
        new_mat.name = f"jewelry_{component_name}"
        new_mat.pbrMetallicRoughness = pygltflib.PbrMetallicRoughness()

        # Set defaults based on component type
        if component_name in ("metal", "prong", "setting", "bail", "clasp"):
            new_mat.pbrMetallicRoughness.baseColorFactor = [0.85, 0.85, 0.87, 1.0]
            new_mat.pbrMetallicRoughness.metallicFactor = 1.0
            new_mat.pbrMetallicRoughness.roughnessFactor = 0.1
        else:  # gemstone
            new_mat.pbrMetallicRoughness.baseColorFactor = [0.98, 0.98, 1.0, 1.0]
            new_mat.pbrMetallicRoughness.metallicFactor = 0.0
            new_mat.pbrMetallicRoughness.roughnessFactor = 0.0

        gltf.materials.append(new_mat)
        return len(gltf.materials) - 1

    @staticmethod
    def _apply_pbr_properties(material, mat_dict: dict):
        """Apply PBR metallic-roughness + extensions to a pygltflib Material."""
        import pygltflib

        pbr = mat_dict["pbrMetallicRoughness"]

        # Core PBR
        if material.pbrMetallicRoughness is None:
            material.pbrMetallicRoughness = pygltflib.PbrMetallicRoughness()
        material.pbrMetallicRoughness.baseColorFactor = pbr["baseColorFactor"]
        material.pbrMetallicRoughness.metallicFactor = pbr["metallicFactor"]
        material.pbrMetallicRoughness.roughnessFactor = pbr["roughnessFactor"]
        material.name = mat_dict.get("name", material.name)

        # Alpha mode for transparent gems
        if "alphaMode" in mat_dict:
            material.alphaMode = mat_dict["alphaMode"]

        # Extensions (transmission, IOR, volume)
        extensions = mat_dict.get("extensions", {})
        if extensions:
            if material.extensions is None:
                material.extensions = {}
            material.extensions.update(extensions)

    @staticmethod
    def _register_extensions(gltf, mat_dict: dict):
        """Register used glTF extensions in the root extensionsUsed array."""
        extensions = mat_dict.get("extensions", {})
        for ext_name in extensions:
            if ext_name not in (gltf.extensionsUsed or []):
                if gltf.extensionsUsed is None:
                    gltf.extensionsUsed = []
                gltf.extensionsUsed.append(ext_name)

    @staticmethod
    def _assign_material_to_component(gltf, mat_index: int, target_component: str):
        """
        Assign material to all mesh primitives belonging to a semantic component.
        Relies on node names or extras containing the component label.
        """
        for node in gltf.nodes:
            is_target = False

            # Check node name
            if node.name and target_component in node.name.lower():
                is_target = True

            # Check node extras
            if node.extras and node.extras.get("semantic_label") == target_component:
                is_target = True

            if is_target and node.mesh is not None:
                mesh = gltf.meshes[node.mesh]
                for prim in mesh.primitives:
                    prim.material = mat_index

    @staticmethod
    def _expand_label_ranges(vertex_labels: dict, n_verts: int) -> dict:
        """
        Expand compressed label ranges to a vertex-index → label mapping.
        Input:  {"metal": [[0, 100], [200, 300]], "gemstone": [[101, 199]]}
        Output: {0: "metal", 1: "metal", ..., 101: "gemstone", ...}
        """
        vert_to_label = {}

        # Handle both dict-of-ranges and list-of-range-dicts formats
        if isinstance(vertex_labels, dict):
            for label, ranges in vertex_labels.items():
                if isinstance(ranges, list):
                    for r in ranges:
                        if isinstance(r, list) and len(r) == 2:
                            for idx in range(r[0], r[1] + 1):
                                vert_to_label[idx] = label
                elif isinstance(ranges, bool):
                    continue  # Skip boolean flags like "metal": True

        elif isinstance(vertex_labels, list):
            # [{start: 0, end: 100, label: "metal"}, ...]
            for entry in vertex_labels:
                label = entry["label"]
                for idx in range(entry["start"], entry["end"] + 1):
                    vert_to_label[idx] = label

        return vert_to_label

    @staticmethod
    def _gltf_to_bytes(gltf) -> bytes:
        """Serialize a pygltflib GLTF2 object to GLB bytes."""
        import pygltflib
        # Convert all buffers to binary blob format for GLB
        gltf.convert_buffers(pygltflib.BufferFormat.BINARYBLOB)
        return b"".join(gltf.save_to_bytes())
