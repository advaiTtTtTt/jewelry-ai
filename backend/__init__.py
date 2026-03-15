"""Compatibility namespace for src-based backend package."""
from pathlib import Path
import pkgutil

__path__ = pkgutil.extend_path(__path__, __name__)
_src_backend = Path(__file__).resolve().parents[1] / "src" / "backend"
if _src_backend.exists():
    __path__.append(str(_src_backend))
