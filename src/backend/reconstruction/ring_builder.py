import trimesh
import numpy as np

class RingBuilder:
    def __init__(self):
        pass
        
    def build_ring(self, params: dict) -> list[trimesh.Trimesh]:
        """
        Builds a mathematically perfect Parametric CAD ring from Gemini blueprint params.
        """
        meshes = []
        
        # 1. Expand parameters into mathematical scale with demo-safe minimums
        b_width = max(params.get("band_width_mm", 3.0), 3.0) / 10.0
        b_thick = max(params.get("band_thickness_mm", 2.5), 2.5) / 10.0
        r_inner = max(params.get("inner_radius_mm", 8.5), 8.5) / 10.0
        r_outer = r_inner + b_thick
        
        # Create perfect CAD annulus (washer shape)
        band = trimesh.creation.annulus(
            r_min=r_inner,
            r_max=r_outer,
            height=b_width,
            sections=128  # High resolution for smoothness
        )
        
        # Annulus extrudes along Z. We rotate it 90 degrees around X so it "stands up" like a ring.
        # Now the hole goes along the Y axis. Top of the ring is at Z = r_outer.
        rot = trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
        band.apply_transform(rot)
        
        # Flatten the bottom slightly for a standard "comfort-fit" feel
        vertices = band.vertices
        # Apply a smooth D-shape or round profile if requested
        if params.get("band_profile") in ["D-shape", "round"]:
            # Simple taubin smoothing rounds the harsh flat edges of the default annulus
            try:
                trimesh.smoothing.filter_taubin(band, iterations=30)
            except Exception:
                pass
                
        band.metadata = {"semantic_label": "metal"}
        meshes.append(band)
        
        # 2. Procedural Gemstone Placement
        if params.get("has_gemstone", True):
            gem_cut = params.get("gem_cut", "round_brilliant")
            gem_r_raw = params.get("gem_radius_mm", 3.5) / 10.0
            max_gem_r = r_outer * 0.4
            gem_r = min(gem_r_raw, max_gem_r)
            
            # Gem sits right on top of the band outer surface
            gem_z = r_outer + gem_r * 0.6
            
            if gem_cut.lower() == "princess":
                # Princess cut = simple bevel box
                gem = trimesh.creation.box([gem_r*2, gem_r*2, gem_r*1.2])
            else:
                # Default Round Brilliant (procedural double cone)
                crown = trimesh.creation.cone(radius=gem_r, height=gem_r*0.5, sections=32)
                pavilion = trimesh.creation.cone(radius=gem_r, height=gem_r*0.8, sections=32)
                # Flip pavilion down
                pavilion.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [1,0,0]))
                pavilion.apply_translation([0, 0, -gem_r*0.8])
                gem = trimesh.util.concatenate([crown, pavilion])
                gem = trimesh.convex.convex_hull(gem)
                
            gem.apply_translation([0, 0, gem_z])
            gem.metadata = {"semantic_label": "gemstone"}
            meshes.append(gem)
            
            # 3. Procedural Prongs
            prong_cnt = params.get("prong_count", 4)
            if prong_cnt > 0:
                for i in range(prong_cnt):
                    angle = (2 * np.pi * i) / prong_cnt
                    
                    # Spread prongs evenly around the gem's circumference
                    px = gem_r * 0.9 * np.cos(angle)
                    py = gem_r * 0.9 * np.sin(angle)
                    
                    prong_r = gem_r * 0.12
                    prong_h = gem_r * 0.8
                    prong = trimesh.creation.cylinder(radius=prong_r, height=prong_h, sections=12)
                    
                    # Position prong base at gem girdle height, pointing upward
                    prong.apply_translation([px, py, gem_z + gem_r * 0.1])
                    
                    # Tilt prong slightly inwards toward gem center
                    tilt_axis = np.array([py, -px, 0.0], dtype=float)
                    norm = np.linalg.norm(tilt_axis)
                    if norm > 1e-8:
                        tilt_axis = tilt_axis / norm
                        tilt_rot = trimesh.transformations.rotation_matrix(
                            0.18,
                            tilt_axis,
                            [px, py, gem_z + gem_r * 0.1],
                        )
                        prong.apply_transform(tilt_rot)
                    
                    prong.metadata = {"semantic_label": "prong"}
                    meshes.append(prong)
                    
        return meshes
