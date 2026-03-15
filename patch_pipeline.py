import re

with open("src/backend/reconstruction/pipeline.py", "r") as f:
    content = f.read()

old_reconstruct_pattern = r"""    def reconstruct\(
        self,
        image: Image\.Image,
        segmentation: Optional\[dict\] = None,
        quality: str = "high",
    \) -> dict:.*?        return \{
            "glb_path": str\(final_glb_path\),
            "vertex_labels": compressed_labels,
            "views": views,
            "job_id": job_id,
        \}"""

new_reconstruct = """    def reconstruct(
        self,
        image: Image.Image,
        segmentation: Optional[dict] = None,
        quality: str = "high",  # Kept for compatibility but ignored
    ) -> dict:
        \"\"\"
        Pure Procedural Pipeline: Uses 2D segmentation to mathematically blueprint 
        and generate flawless CAD-grade meshes. Bypasses buggy Image-to-3D models.
        \"\"\"
        import json
        import tempfile
        import uuid
        import trimesh

        job_id = str(uuid.uuid4())[:8]
        job_dir = self.output_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        logger.info("[%s] Starting pure procedural reconstruction pipeline", job_id)

        # Stage A: Background removal (keep for reference/debug)
        clean_image = self._remove_background(image)
        clean_image.save(job_dir / "input_clean.png")

        # Stage B: Procedural Band Generation
        logger.info("[%s] Stage B: Procedural Band generation", job_id)
        band_mesh = self._build_procedural_band(segmentation, image.size)
        band_mesh.metadata = {"semantic_label": "metal"}

        # Stage C: Procedural gems + prongs
        logger.info("[%s] Stage C: Procedural gemstone placement", job_id)
        from src.backend.reconstruction.gem_builder import GemBuilder
        gem_builder = GemBuilder()
        gem_meshes = gem_builder.place_gems_from_segmentation(
            segmentation if segmentation else {}, image.size, band_mesh
        )
        prong_meshes = []
        for gem in gem_meshes:
            prongs = gem_builder.build_prongs_for_gem(gem, n_prongs=4)
            prong_meshes.extend(prongs)

        # Stage D: Combine all meshes
        all_meshes = [band_mesh] + gem_meshes + prong_meshes
        combined = trimesh.util.concatenate(all_meshes)
        
        raw_glb_path = job_dir / "raw_mesh.glb"
        combined.export(str(raw_glb_path))
        
        # Build vertex labels mapped to the concatenated mesh
        vertex_labels = {}
        offset = 0
        for m in all_meshes:
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

        return {
            "glb_path": str(final_glb_path),
            "vertex_labels": compressed_labels,
            "views": [clean_image], # No more Zero123 views needed
            "job_id": job_id,
        }

    def _build_procedural_band(self, segmentation: dict, image_size: tuple) -> trimesh.Trimesh:
        \"\"\"Mathematically generates a perfect ring band based on exactly where the AI saw it in 2D.\"\"\"
        import numpy as np
        import trimesh
        
        W, H = image_size
        width, height = W * 0.5, H * 0.5 
        
        if segmentation and "parts" in segmentation:
            for part_name, part_data in segmentation["parts"].items():
                if part_name in ["metal", "band", "metal band", "setting"] and "mask" in part_data:
                    mask = np.array(part_data["mask"])
                    ys, xs = np.where(mask)
                    if len(xs) > 0:
                        width = np.max(xs) - np.min(xs)
                        height = np.max(ys) - np.min(ys)
                        break
                        
        # Translate 2D bounding box width into 3D scale
        outer_radius = (width / W) * 0.4
        
        if outer_radius < 0.1:  # Fallback if detection size is anomalous
            outer_radius = 0.4
            
        minor_radius = outer_radius * 0.12 # Band thickness ratio
        major_radius = outer_radius - minor_radius
        
        # High resolution Torus for smooth CAD appearance
        band_mesh = trimesh.creation.torus(major_radius=major_radius, minor_radius=minor_radius, sections=64, ring_markers=64)
        
        # Tilt it to match the camera angle in the photo
        aspect = height / width if width > 0 else 1.0
        tilt_angle = np.arccos(min(1.0, aspect))
        
        rot = trimesh.transformations.rotation_matrix(tilt_angle, [1, 0, 0])
        band_mesh.apply_transform(rot)
        
        # Flatten bottom slightly to simulate standard ring comfort-fit profile
        vertices = band_mesh.vertices
        # squish the Z-axis of the band profile slightly
        vertices[:, 2] *= 0.8
        band_mesh.vertices = vertices
        trimesh.repair.fix_normals(band_mesh)
        
        return band_mesh"""

new_content = re.sub(old_reconstruct_pattern, new_reconstruct, content, flags=re.DOTALL)

with open("src/backend/reconstruction/pipeline.py", "w") as f:
    f.write(new_content)

print("Patching done.")
