with open("backend/reconstruction/pipeline.py", "r") as f:
    text = f.read()

import_target = "    def reconstruct("
end_target = "    # ═══════════════════════════════════════════════════════════════════\n    # STAGE A"

start_idx = text.find(import_target)
end_idx = text.find(end_target)

reconstruct_func = """    def reconstruct(
        self,
        image: Image.Image,
        segmentation: Optional[dict] = None,
    ) -> dict:
        \"\"\"
        Hybrid pipeline: TripoSR handles the band/shank. 
        Gemstones and prongs are procedurally generated from segmentation data.
        \"\"\"
        job_id = str(uuid.uuid4())[:8]
        job_dir = self.output_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        logger.info("[%s] Starting hybrid reconstruction pipeline", job_id)

        # Stage A: Remove background
        logger.info("[%s] Stage A: Background removal", job_id)
        clean_image = self._remove_background(image)
        clean_image.save(job_dir / "input_clean.png")

        # Stage 1: Band-only masking
        band_image = self._mask_to_band_only(clean_image, segmentation if segmentation else {})
        band_image.save(job_dir / "input_band_only.png")

        # Stage B: Generate multi-view images via Zero123++
        logger.info("[%s] Stage B: Multi-view generation (Zero123++) on band", job_id)
        views = self._generate_multiviews(band_image)
        for i, view in enumerate(views):
            view.save(job_dir / f"view_{i}_{ZERO123_VIEWS[i]['name']}.png")

        # Stage C: Reconstruct 3D band via TripoSR
        logger.info("[%s] Stage C: 3D band reconstruction (TripoSR)", job_id)
        images_to_triposr = [self._prepare_for_triposr(v) for v in views]
        if not images_to_triposr:
            images_to_triposr = [self._prepare_for_triposr(band_image)]
            
        for i, prepared in enumerate(images_to_triposr):
            prepared.save(job_dir / f"triposr_input_debug_{i}.png")

        print(f"Feeding {len(images_to_triposr)} band images to TripoSR")
        band_mesh = self._reconstruct_mesh(images_to_triposr)
        band_mesh.metadata = {"semantic_label": "metal"}

        # Stage D: Procedural gems + prongs
        logger.info("[%s] Stage D: Procedural gemstone placement", job_id)
        gem_builder = GemBuilder()
        gem_meshes = gem_builder.place_gems_from_segmentation(
            segmentation if segmentation else {}, image.size, band_mesh
        )
        prong_meshes = []
        for gem in gem_meshes:
            prongs = gem_builder.build_prongs_for_gem(gem, n_prongs=4)
            prong_meshes.extend(prongs)

        # Stage 4: Combine all meshes
        all_meshes = [band_mesh] + gem_meshes + prong_meshes
        
        # We concatenate instead of scene so vertices are unified for applier.py
        combined = trimesh.util.concatenate(all_meshes)
        
        # Export raw merged GLB
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

        logger.info("[%s] Pipeline complete -> %s", job_id, final_glb_path)

        return {
            "glb_path": str(final_glb_path),
            "vertex_labels": compressed_labels,
            "views": views,
            "job_id": job_id,
        }

"""

new_text = text[:start_idx] + reconstruct_func + text[end_idx:]
with open("backend/reconstruction/pipeline.py", "w") as f:
    f.write(new_text)

