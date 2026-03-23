try:
    from .docling_backend import DoclingBackend
except ImportError:
    DoclingBackend = None

from .lighton_backend import LightOnOCRBackend

__all__ = [
    "DoclingBackend",
    "LightOnOCRBackend",
]
