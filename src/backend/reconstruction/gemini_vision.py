import os
import json
import logging
import importlib
from PIL import Image, ImageStat

genai = importlib.import_module("google.generativeai")

logger = logging.getLogger(__name__)

PRIMARY_API_KEY = "AIzaSyC8REiPn1ZXkGPkh9okjnkElyPg9wRrGR4"
FALLBACK_API_KEY = "AIzaSyCgpD7LeiY-nqhGti97r6sRggseUyvufqk"
FALLBACK_API_KEY_2 = "AIzaSyAoxProkm_6tmfUctuZ4MIsgEK7vXcg-wA"
FALLBACK_API_KEY_3 = "AIzaSyCGBqs1esZOIi_YOmBx7KojiWrDOLTUNc4"


def _load_env_file(path: str = ".env") -> None:
    if not os.path.exists(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as env_file:
            for line in env_file:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                if "=" not in stripped:
                    continue
                key, value = stripped.split("=", 1)
                if key and key not in os.environ:
                    os.environ[key] = value
    except Exception as error:
        logger.warning("Could not load .env file: %s", error)


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


def _get_average_brightness(image: Image.Image) -> float:
    gray = image.convert("L")
    stat = ImageStat.Stat(gray)
    return float(stat.mean[0])  # 0-255 range


def _heuristic_params(image: Image.Image) -> dict:
    # Simple heuristic using image properties; avoids any API usage
    width, height = image.size
    brightness = _get_average_brightness(image)

    # Adjust gem size slightly based on brightness; keep within reasonable bounds
    gem_radius = 3.5
    if brightness < 70:
        gem_radius = 3.2
    elif brightness > 180:
        gem_radius = 3.8

    # Wider bands for very bright/large images to stay visually balanced
    band_width = 3.5 if max(width, height) < 2000 else 4.0
    band_thickness = 2.5 if brightness > 60 else 2.8

    return {
        "band_profile": "D-shape",
        "band_width_mm": band_width,
        "band_thickness_mm": band_thickness,
        "inner_radius_mm": 8.5,
        "has_gemstone": True,
        "gem_cut": "round_brilliant",
        "gem_radius_mm": gem_radius,
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

def analyze_jewelry_image(image: Image.Image, skip_api: bool = False) -> dict:
    """
    Uses Gemini 1.5 Flash to extract structural CAD parameters from a jewelry image.
    Returns a dictionary of exact parameters to drive the procedural builder.
    """
    _load_env_file()
    logger.info("Executing Gemini Vision parameter extraction...")

    env_skip = os.getenv("SKIP_GEMINI_API", "").strip().lower() in {"1", "true", "yes", "on"}
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    skip_reasons = []
    if skip_api:
        skip_reasons.append("skip_api=True")
    if env_skip:
        skip_reasons.append("SKIP_GEMINI_API env flag")
    if not api_key:
        skip_reasons.append("missing GEMINI_API_KEY")

    if skip_api or env_skip or not api_key:
        print(f"GEMINI: skipping API ({'; '.join(skip_reasons)}) -> using heuristic defaults")
        return _heuristic_params(image)

    print(
        "GEMINI KEY STATUS: {} starts with: {}".format(
            "SET",
            api_key[:8],
        )
    )
    
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
    print(
        "GEMINI KEY CANDIDATES ({}): {}".format(
            len(keys), [k[:6] + "..." for k in keys]
        )
    )
    if not keys:
        logger.error("No Gemini API key found. Falling back to heuristic parameters.")
        print("GEMINI: no keys available -> heuristic fallback")
        return _heuristic_params(image)

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
        "All Gemini keys failed (%s). Falling back to heuristic parameters.",
        last_error,
    )
    print(f"GEMINI ERROR: {type(last_error).__name__}: {last_error}")
    print("GEMINI: using heuristic fallback after failures")
    return _heuristic_params(image)
