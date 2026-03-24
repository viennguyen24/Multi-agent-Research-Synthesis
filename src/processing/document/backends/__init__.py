try:
    from .docling_backend import DoclingBackend
except ImportError:
    DoclingBackend = None

try:
    from .chandra_backend import ChandraOCRBackend
except ImportError:
    ChandraOCRBackend = None

try:
    from .glm_backend import GLMOCRBackend
except ImportError:
    GLMOCRBackend = None

try:
    from .lighton_backend import LightOnOCRBackend
except ImportError:
    LightOnOCRBackend = None

__all__ = [
    "DoclingBackend",
    "ChandraOCRBackend",
    "LightOnOCRBackend",
    "GLMOCRBackend",
]
