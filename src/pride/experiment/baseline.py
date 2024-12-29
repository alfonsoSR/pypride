from typing import TYPE_CHECKING
from ..logger import log
from astropy import time

if TYPE_CHECKING:
    from .experiment import Experiment
    from .station import Station
    from .observation import Observation


class Baseline:
    """VLBI baseline

    A baseline is a pair of stations that have observations of different sources associated with them

    :param center: Station object representing the phase center
    :param station: Station object representing the other station in the baseline
    :param observations: List of observations associated with the baseline
    :param nobs: Number of observations associated with the baseline
    :param tstamps: Time stamps of the observations associated with the baseline
    :param exp: Experiment object to which the baseline belongs
    """

    def __init__(self, center: "Station", station: "Station") -> None:

        self.center = center
        self.station = station
        self.observations: list["Observation"] = []

        # Optional attributes
        setattr(self, "_exp", getattr(station, "_exp"))
        setattr(self, "__tstamps", None)

        return None

    @property
    def exp(self) -> "Experiment":

        if getattr(self, "_exp") is None:
            log.error(
                "Experiment not set for"
                f" {self.center.name}-{self.station.name} baseline"
            )
            exit(1)
        return getattr(self, "_exp")

    @property
    def id(self) -> str:
        return f"{self.center.name}-{self.station.name}"

    def __str__(self) -> str:
        return self.id

    @property
    def tstamps(self) -> "time.Time":

        if getattr(self, "__tstamps") is None:
            log.error(f"Time stamps not found for baseline {self}")
            exit(0)
        return getattr(self, "__tstamps")

    @property
    def nobs(self) -> int:
        return len(self.observations)

    def add_observation(self, observation: "Observation") -> None:
        """Add observation to baseline"""

        self.observations.append(observation)
        return None

    def update_time_stamps(self) -> None:

        _tstamps = time.Time(
            [observation.tstamps for observation in self.observations],
            scale="utc",
        ).sort()
        setattr(self, "__tstamps", _tstamps)

        return None
