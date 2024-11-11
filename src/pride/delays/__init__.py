from .core import Delay
from .models import Geometric, Tropospheric, Ionospheric, ThermalDeformation

DELAY_MODELS: dict[str, type["Delay"]] = {
    Geometric.name: Geometric,
    Tropospheric.name: Tropospheric,
    Ionospheric.name: Ionospheric,
    ThermalDeformation.name: ThermalDeformation,
}

__all__ = [
    "Delay",
    "DELAY_MODELS",
    "Geometric",
    "Tropospheric",
    "Ionospheric",
    "ThermalDeformation",
]
