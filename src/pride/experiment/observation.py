from typing import TYPE_CHECKING
import datetime
from astropy import time
from ..logger import log

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

        self.source = source
        self.band = band
        _tstamps = time.Time(sorted(tstamps), scale="utc")
        self.tstamps = time.Time(
            sorted(tstamps),
            scale="utc",
            location=baseline.station.tectonic_corrected_location(_tstamps),
        )
        self.baseline = baseline

        # Optional attributes
        setattr(self, "_exp", getattr(baseline.station, "_exp"))

        return None

    @property
    def exp(self) -> "Experiment":

        if getattr(self, "_exp") is None:
            log.error(f"Experiment not set for {self.source.name} observation")
            exit(1)
        return getattr(self, "_exp")
