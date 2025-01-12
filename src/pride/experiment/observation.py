from typing import TYPE_CHECKING, Any
import datetime
from astropy import time
from ..logger import log
import numpy as np

if TYPE_CHECKING:
    from .experiment import Experiment
    from .source import Source
    from ..types import Band
    from .baseline import Baseline


class Observation:
    """Observation of a source

    The time stamps are stored as an astropy Time object including information about the coordinates of the station for which the observation was performed. The coordinates of the station are obtained by correcting the location at a reference epoch (loaded from data file) for tectonic motion.

    :param source: Source object representing the target
    :param band: Frequency band in which the source was detected
    :param tstamps: Collection of epochs in which the source was detected
    """

    __slots__ = (
        "source",
        "band",
        "tstamps",
        "baseline",
        "station",
        "exp",
        "icrf2itrf",
        "seu2itrf",
        "dot_icrf2itrf",
    )

    def __init__(
        self,
        baseline: "Baseline",
        source: "Source",
        band: "Band",
        tstamps: list[datetime.datetime],
    ) -> None:
        """Initialize observation

        :param Baseline: Baseline from which the observation was performed
        :param source: Source object representing the target
        :param band: Frequency band in which the source was detected
        :param tstamps: List of UTC epochs in which the source was detected
        :param exp: Experiment object to which the observation belongs
        """

        # Input-based properties
        self.source = source
        self.band = band
        _tstamps = time.Time(sorted(tstamps), scale="utc")
        self.tstamps = time.Time(
            _tstamps,
            scale="utc",
            location=baseline.station.tectonic_corrected_location(_tstamps),
        )
        self.baseline = baseline
        self.station = baseline.station

        # Optional properties
        self.exp: "Experiment" = NotImplemented
        self.icrf2itrf: "np.ndarray" = NotImplemented
        self.seu2itrf: "np.ndarray" = NotImplemented
        self.dot_icrf2itrf: "np.ndarray" = NotImplemented

        return None

    def __getattribute__(self, name: str) -> Any:

        val = super().__getattribute__(name)
        if val is NotImplemented:
            log.error(f"Attribute {name} is not set for observation")
            exit(1)
        return val
