from .core import Displacement
from .models import SolidTide, OceanLoading, PoleTide

DISPLACEMENT_MODELS: dict[str, type["Displacement"]] = {
    SolidTide.name: SolidTide,
    OceanLoading.name: OceanLoading,
    PoleTide.name: PoleTide,
}

__all__ = [
    "Displacement",
    "DISPLACEMENT_MODELS",
    "SolidTide",
]
