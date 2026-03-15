import re

with open("src/backend/reconstruction/pipeline.py", "r") as f:
    orig = f.read()

new_reconstruct = """    def reconstruct(
        self,
        image: Image.Image,
        segmentation: Optional[dict] = None,
        quality: str = "high",
    ) -> dict:
        \"\"\"
        Pure Procedural Pipeline: Uses Gemini VLM to extract a mathematical blueprint, 
        and generates flawless CAD-grade parametric meshes.
        \"\"\"
        import json
        import tempfile
        import uuid
        import trimesh

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
    # STAGE A: Background Removal"""

# Using regex to replace the old reconstruct all the way up to STAGE A
pattern = re.compile(r'    def reconstruct\([\s\S]*?# ═══════════════════════════════════════════════════════════════════\n    # STAGE A: Background Removal', re.MULTILINE)

new_content = pattern.sub(new_reconstruct, orig)

with open("src/backend/reconstruction/pipeline.py", "w") as f:
    f.write(new_content)

print("Pipeline patched successfully.")
