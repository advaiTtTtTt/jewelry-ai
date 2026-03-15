"""
PBR Material Definitions for Jewelry
=====================================
Full physically-based rendering (PBR) properties for metals and gemstones.

Metals use the metallic-roughness workflow (glTF 2.0 core spec).
Gemstones additionally use glTF extensions:
  - KHR_materials_transmission  (transparency)
  - KHR_materials_ior           (index of refraction)
  - KHR_materials_volume        (colored absorption, like ruby's red tint)

These extensions are auto-mapped by Three.js GLTFLoader to
MeshPhysicalMaterial properties (transmission, ior, thickness, etc.)
"""

# ═══════════════════════════════════════════════════════════════════════
# METALS — metallic-roughness PBR
# ═══════════════════════════════════════════════════════════════════════
# color: [R, G, B] in 0.0–1.0 linear sRGB
# metallic: always 1.0 for metals
# roughness: polished metals ≈ 0.02–0.15

METALS = {
    "yellow_gold": {
        "name": "Yellow Gold (18K)",
        "color": [1.0, 0.766, 0.336],       # Warm gold
        "metallic": 1.0,
        "roughness": 0.1,
        "hex": "#FFC355",                    # For frontend swatches
    },
    "white_gold": {
        "name": "White Gold (18K)",
        "color": [0.85, 0.85, 0.87],         # Cool silver-white
        "metallic": 1.0,
        "roughness": 0.05,
        "hex": "#D9D9DE",
    },
    "rose_gold": {
        "name": "Rose Gold (18K)",
        "color": [0.91, 0.69, 0.59],         # Pinkish copper
        "metallic": 1.0,
        "roughness": 0.1,
        "hex": "#E8B096",
    },
    "platinum": {
        "name": "Platinum",
        "color": [0.83, 0.83, 0.85],         # Very white/grey
        "metallic": 1.0,
        "roughness": 0.02,                    # Highly polished
        "hex": "#D4D4D9",
    },
    "silver": {
        "name": "Sterling Silver",
        "color": [0.78, 0.78, 0.80],         # Slightly darker than platinum
        "metallic": 1.0,
        "roughness": 0.15,
        "hex": "#C7C7CC",
    },
}

# ═══════════════════════════════════════════════════════════════════════
# GEMSTONES — transmission + IOR + volume absorption
# ═══════════════════════════════════════════════════════════════════════
# color: [R, G, B] base color (affects both surface and volume)
# ior: index of refraction (diamond=2.42, glass=1.5)
# transmission: 0=opaque, 1=fully transparent (gem-like)
# roughness: surface micro-roughness (polished gems ≈ 0.0)
# attenuation_color: volumetric color absorption (light passing through)
# attenuation_distance: how far light travels before full absorption
# thickness: estimated gem thickness in scene units

GEMSTONES = {
    "diamond": {
        "name": "Diamond",
        "color": [0.98, 0.98, 1.0],          # Near-white with blue tint
        "ior": 2.42,                           # Highest IOR — causes "fire"
        "transmission": 0.95,
        "roughness": 0.0,
        "attenuation_color": [1.0, 1.0, 1.0], # No color absorption
        "attenuation_distance": 100.0,         # Light travels very far
        "thickness": 0.5,
        "dispersion": 0.044,                   # Chromatic dispersion ("fire")
        "hex": "#F8F8FF",
    },
    "ruby": {
        "name": "Ruby",
        "color": [0.88, 0.07, 0.14],          # Deep red
        "ior": 1.77,
        "transmission": 0.7,
        "roughness": 0.0,
        "attenuation_color": [0.9, 0.05, 0.1], # Red absorption
        "attenuation_distance": 2.0,
        "thickness": 0.4,
        "dispersion": 0.018,
        "hex": "#E01224",
    },
    "sapphire": {
        "name": "Sapphire",
        "color": [0.06, 0.12, 0.55],          # Deep blue
        "ior": 1.77,
        "transmission": 0.7,
        "roughness": 0.0,
        "attenuation_color": [0.05, 0.1, 0.5], # Blue absorption
        "attenuation_distance": 2.0,
        "thickness": 0.4,
        "dispersion": 0.018,
        "hex": "#0F1F8C",
    },
    "emerald": {
        "name": "Emerald",
        "color": [0.10, 0.60, 0.25],          # Rich green
        "ior": 1.58,
        "transmission": 0.6,
        "roughness": 0.02,                     # Slight inclusions
        "attenuation_color": [0.08, 0.55, 0.2],
        "attenuation_distance": 1.5,
        "thickness": 0.4,
        "dispersion": 0.014,
        "hex": "#1A993F",
    },
    "amethyst": {
        "name": "Amethyst",
        "color": [0.55, 0.20, 0.70],          # Purple-violet
        "ior": 1.54,
        "transmission": 0.65,
        "roughness": 0.01,
        "attenuation_color": [0.5, 0.18, 0.65],
        "attenuation_distance": 3.0,
        "thickness": 0.5,
        "dispersion": 0.013,
        "hex": "#8C33B3",
    },
    "cubic_zirconia": {
        "name": "Cubic Zirconia",
        "color": [0.95, 0.95, 0.97],          # Near-colorless
        "ior": 2.15,
        "transmission": 0.9,
        "roughness": 0.0,
        "attenuation_color": [1.0, 1.0, 1.0], # Colorless like diamond
        "attenuation_distance": 80.0,
        "thickness": 0.5,
        "dispersion": 0.058,                   # More fire than diamond
        "hex": "#F2F2F7",
    },
}


def get_all_materials() -> dict:
    """Return all materials organized by category."""
    return {
        "metals": METALS,
        "gemstones": GEMSTONES,
    }


def get_material(name: str) -> dict:
    """Look up a material by name. Checks both metals and gemstones."""
    if name in METALS:
        return {"type": "metal", **METALS[name]}
    if name in GEMSTONES:
        return {"type": "gemstone", **GEMSTONES[name]}
    raise ValueError(f"Unknown material: '{name}'. Available: {list(METALS) + list(GEMSTONES)}")


def build_gltf_material_dict(material_name: str) -> dict:
    """
    Build a glTF 2.0-compatible material dict including PBR extensions.

    This dict can be used directly with pygltflib to create/update a
    glTF Material object with proper extension data.
    """
    mat = get_material(material_name)

    # Base PBR metallic-roughness (glTF core spec)
    gltf_mat = {
        "name": mat["name"],
        "pbrMetallicRoughness": {
            "baseColorFactor": mat["color"] + [1.0],  # RGBA
            "metallicFactor": mat.get("metallic", 0.0),
            "roughnessFactor": mat.get("roughness", 0.5),
        },
        "extensions": {},
    }

    # For gemstones: add transmission, IOR, and volume extensions
    if mat["type"] == "gemstone":
        gltf_mat["pbrMetallicRoughness"]["metallicFactor"] = 0.0  # Gems aren't metallic
        gltf_mat["alphaMode"] = "BLEND"

        # KHR_materials_transmission — makes the gem transparent
        gltf_mat["extensions"]["KHR_materials_transmission"] = {
            "transmissionFactor": mat["transmission"],
        }

        # KHR_materials_ior — refraction index
        gltf_mat["extensions"]["KHR_materials_ior"] = {
            "ior": mat["ior"],
        }

        # KHR_materials_volume — volumetric color absorption
        gltf_mat["extensions"]["KHR_materials_volume"] = {
            "thicknessFactor": mat.get("thickness", 0.5),
            "attenuationColor": mat.get("attenuation_color", [1, 1, 1]),
            "attenuationDistance": mat.get("attenuation_distance", 10.0),
        }

    return gltf_mat
