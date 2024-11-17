from typing import Any, TYPE_CHECKING
from ..logger import log
from abc import abstractmethod

if TYPE_CHECKING:
    from ..experiment.core import Experiment, Station
    from astropy import time


class Displacement:
    """Base class for implementation of geophysical displacements

    Used to correct the position of stations for geophysical phenomena
    """

    name: str = NotImplemented
    etc: dict[str, Any] = NotImplemented
    requires_spice: bool = NotImplemented
    models: list[str] = NotImplemented

    def __init__(self, experiment: "Experiment") -> None:

        self.exp = experiment
        self.config: dict[str, Any] = self.exp.setup.displacements[self.name]
        self._resources: dict[str, Any] = {}
        self.resources: dict[str, Any] = {}
        self.ensure_resources()
        log.debug(f"Acquired resources for {self.name} displacements")
        return None

    @abstractmethod
    def ensure_resources(self) -> None:
        log.error(
            f"Method ensure_resources not implemented for {self.name} displacement"
        )
        exit(1)

    @abstractmethod
    def load_resources(self) -> None:
        log.error(f"Method load_resources not implemented for {self.name} displacement")
        exit(1)

    @abstractmethod
    def calculate(self, station: "Station", epoch: "time.Time"):

        log.error(f"Method calculate not implemented for {self.name} displacement")
        exit(1)
