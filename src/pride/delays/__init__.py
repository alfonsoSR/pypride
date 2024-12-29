from .models import Geometric, Tropospheric, Ionospheric, ThermalDeformation
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .core import Delay

DELAY_MODELS: dict[str, type["Delay"]] = {
    Geometric.name: Geometric,
    Tropospheric.name: Tropospheric,
    Ionospheric.name: Ionospheric,
    ThermalDeformation.name: ThermalDeformation,
}

__all__ = ["DELAY_MODELS"]
