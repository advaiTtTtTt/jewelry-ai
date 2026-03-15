import os
import json
import logging
import importlib
from PIL import Image

genai = importlib.import_module("google.generativeai")

logger = logging.getLogger(__name__)

PRIMARY_API_KEY = "AIzaSyC8REiPn1ZXkGPkh9okjnkElyPg9wRrGR4"
FALLBACK_API_KEY = "AIzaSyCgpD7LeiY-nqhGti97r6sRggseUyvufqk"
FALLBACK_API_KEY_2 = "AIzaSyAoxProkm_6tmfUctuZ4MIsgEK7vXcg-wA"
FALLBACK_API_KEY_3 = "AIzaSyCGBqs1esZOIi_YOmBx7KojiWrDOLTUNc4"


def _default_params() -> dict:
    return {
        "band_profile": "D-shape",
        "band_width_mm": 2.5,
        "band_thickness_mm": 1.5,
        "inner_radius_mm": 8.0,
        "has_gemstone": True,
        "gem_cut": "round_brilliant",
        "gem_radius_mm": 3.0,
        "prong_count": 4,
    }


def _candidate_keys() -> list[str]:
    keys = [
        os.getenv("GEMINI_API_KEY", ""),
        os.getenv("GEMINI_API_KEY_FALLBACK", ""),
        os.getenv("GEMINI_API_KEY_FALLBACK_2", ""),
        os.getenv("GEMINI_API_KEY_FALLBACK_3", ""),
        PRIMARY_API_KEY,
        FALLBACK_API_KEY,
        FALLBACK_API_KEY_2,
        FALLBACK_API_KEY_3,
    ]
    unique_keys: list[str] = []
    for key in keys:
        if key and key not in unique_keys:
            unique_keys.append(key)
    return unique_keys


def _is_quota_error(error: Exception) -> bool:
    message = str(error).lower()
    return any(
        token in message
        for token in ["quota", "rate", "429", "resource_exhausted", "credit", "billing"]
    )

def analyze_jewelry_image(image: Image.Image) -> dict:
    """
    Uses Gemini 1.5 Flash to extract structural CAD parameters from a jewelry image.
    Returns a dictionary of exact parameters to drive the procedural builder.
    """
    logger.info("Executing Gemini Vision parameter extraction...")
    
    prompt = """
    You are an expert 3D pipeline parametric extractor for jewelry.
    Analyze this image of a ring and extract its exact structural parameters.
    Respond ONLY with a valid JSON object, no markdown formatting or extra text.
    The JSON must follow this exact schema:
    {
        "band_profile": "D-shape",    // either "flat", "D-shape", or "round"
        "band_width_mm": 2.5,         // float, typical is 1.5 to 5.0
        "band_thickness_mm": 1.5,     // float, typical is 1.2 to 2.5
        "inner_radius_mm": 8.0,       // float, typical is 8.0 for a size 6 ring
        "has_gemstone": true,         // boolean
        "gem_cut": "round_brilliant", // "round_brilliant" or "princess"
        "gem_radius_mm": 3.0,         // float
        "prong_count": 4              // integer, usually 0, 4, or 6
    }
    """
    
    # Ensure correct mode for the Vision model
    if image.mode != "RGB":
        image_rgb = image.convert("RGB")
    else:
        image_rgb = image
        
    keys = _candidate_keys()
    if not keys:
        logger.error("No Gemini API key found. Falling back to default parameters.")
        return _default_params()

    last_error: Exception | None = None
    for index, key in enumerate(keys):
        try:
            genai.configure(api_key=key)
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content([prompt, image_rgb])
            text = response.text.replace("```json", "").replace("```", "").strip()
            params = json.loads(text)
            logger.info("Gemini parameter extraction succeeded with key #%d", index + 1)
            return params
        except Exception as error:
            last_error = error
            if _is_quota_error(error) and index < len(keys) - 1:
                logger.warning(
                    "Gemini key #%d exhausted/rate-limited (%s). Trying fallback key...",
                    index + 1,
                    error,
                )
                continue
            logger.warning("Gemini key #%d failed: %s", index + 1, error)

    logger.error(
        "All Gemini keys failed (%s). Falling back to default parameters.",
        last_error,
    )
    return _default_params()
