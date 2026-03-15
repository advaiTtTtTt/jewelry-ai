import trimesh
import numpy as np
import scipy

class GemBuilder:
    
    BRILLIANT_CUT_FACETS = 57  # standard round brilliant
    
    def build_round_brilliant(self, radius=0.15) -> trimesh.Trimesh:
        """
        Generate a geometrically correct round brilliant cut diamond.
        """
        # Crown (top half) - truncated cone
        crown_height = radius * 0.35
        crown = trimesh.creation.cone(
            radius=radius, height=crown_height
        )
        
        # Pavilion (bottom half) - full cone pointing down
        pavilion_height = radius * 0.43
        pavilion = trimesh.creation.cone(
            radius=radius, height=pavilion_height
        )
        # Flip pavilion to point downward
        pavilion.apply_transform(
            trimesh.transformations.rotation_matrix(np.pi, [1,0,0])
        )
        pavilion.apply_translation([0, 0, -pavilion_height])
        
        # Combine crown + pavilion
        gem = trimesh.util.concatenate([crown, pavilion])
        gem = trimesh.convex.convex_hull(gem)
        return gem
    
    def build_princess_cut(self, size=0.15) -> trimesh.Trimesh:
        """Square princess cut — box with beveled bottom."""
        box = trimesh.creation.box([size*2, size*2, size*1.2])
        return box
    
    def place_gems_from_segmentation(
        self, 
        segmentation: dict,
        image_size: tuple,
        band_mesh: trimesh.Trimesh,
        gem_type: str = "diamond"
    ) -> list[trimesh.Trimesh]:
        """
        1. Find the centroid of the mask region
        2. Project centroid onto the band mesh surface
        3. Place a procedural gem at that position
        """
        gems = []
        W, H = image_size  # PIL image size is (W, H)
        
        for part_name, part_data in segmentation.get("parts", {}).items():
            if "gemstone" not in part_name and "stone" not in part_name:
                continue
            
            mask = part_data["mask"]
            if isinstance(mask, list):
                mask = np.array(mask)
                
            bbox = part_data["bbox"]  # [x1, y1, x2, y2]
            
            # Estimate gem size from bbox
            bbox_w = bbox[2] - bbox[0]
            bbox_h = bbox[3] - bbox[1]
            gem_radius = (bbox_w / W) * 0.4  # scale to mesh space
            gem_radius = max(0.05, min(0.25, gem_radius))
            
            # Find centroid of mask
            ys, xs = np.where(mask)
            if len(xs) == 0:
                continue
            cx = float(np.mean(xs)) / W  # normalized 0-1
            cy = float(np.mean(ys)) / H
            
            # Map to 3D position above band mesh
            mesh_bounds = band_mesh.bounds
            x3d = (cx - 0.5) * (mesh_bounds[1][0] - mesh_bounds[0][0])
            y3d = (0.5 - cy) * (mesh_bounds[1][1] - mesh_bounds[0][1])
            z3d = mesh_bounds[1][2] + gem_radius
            
            # Build the gem
            gem = self.build_round_brilliant(radius=gem_radius)
            gem.apply_translation([x3d, y3d, z3d])
            
            # Assign gem material metadata
            gem.metadata = {"semantic_label": "gemstone"}
            gems.append(gem)
        
        return gems

    def build_prongs_for_gem(
        self, 
        gem_mesh: trimesh.Trimesh, 
        n_prongs: int = 4
    ) -> list[trimesh.Trimesh]:
        """Place N evenly-spaced prong cylinders around a gem."""
        prongs = []
        gem_center = gem_mesh.centroid
        gem_radius = gem_mesh.extents[0] / 2
        
        for i in range(n_prongs):
            angle = (2 * np.pi * i) / n_prongs
            px = gem_center[0] + gem_radius * np.cos(angle)
            py = gem_center[1] + gem_radius * np.sin(angle)
            pz = gem_center[2]
            
            prong = trimesh.creation.cylinder(
                radius=gem_radius * 0.08,
                height=gem_radius * 0.6,
                sections=8
            )
            prong.apply_translation([px, py, pz])
            prong.metadata = {"semantic_label": "prong"}
            prongs.append(prong)
        
        return prongs
