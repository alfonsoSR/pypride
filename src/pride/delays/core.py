from abc import abstractmethod
from typing import TYPE_CHECKING, Any
from ..logger import log

if TYPE_CHECKING:
    from ..experiment.experiment import Experiment


class Delay:
    """Base class for implementation of delay models

    :param name: Unique name that identifies the delay model
    :param etc: Configuration parameters not meant to be adjusted by users
    :param requires_spice: Whether the delay model requires SPICE kernels
    :param station_specific: Whether the resources required to calculate the delay are station-specific
    :param exp: Experiment object
    :param config: Section of the configuration file associated with the delay
    :param resources: Private container to be used internally when loading resources
    """

    name: str = NotImplemented
    etc: dict[str, Any] = NotImplemented
    requires_spice: bool = NotImplemented
    station_specific: bool = NotImplemented

    def __init__(self, exp: "Experiment") -> None:
        """Initialize delay model from experiment"""

        self.exp = exp
        self.config: dict[str, Any] = self.exp.setup.delays[self.name]
        self.resources: dict[str, Any] = {}

        # Ensure resources required to calculate the delay
        self.ensure_resources()
        log.debug(f"Acquired resources for {self.name} delay")

        return None

    @abstractmethod
    def ensure_resources(self) -> None:
        """Ensure availability of external resources required to calculate the delay"""

        log.error(
            f"Method ensure_resources not implemented for {self.name} delay"
        )
        exit(1)

        return None

    def load_resources(self) -> dict[str, Any]:
        """Load resources required to calculate the delay

        Updates the baselines of the experiment with resources required to calculate the delay in their specific time domain
        """
        log.error(
            f"Method load_resources not implemented for {self.name} delay"
        )
        exit(1)
