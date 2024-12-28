from .core import Displacement
from typing import Any, TYPE_CHECKING
from ..io import internal_file
from ..logger import log
import numpy as np
from astropy import time
import spiceypy as spice
from ..constants import J2000, Constants
from nastro import types as nt
import erfa

if TYPE_CHECKING:
    from ..experiment.core import Station


class OceanLoading(Displacement):
    """Displacement due to ocean loading"""

    name = "OceanLoading"
    etc = {}
    models = ["tpxo72"]

    def ensure_resources(self) -> None:

        source = self.config["data"] / f"{self.config['model']}.blq"
        if not source.exists():
            log.error(
                f"Failed to initialize {self.name} displacement: {source} not found"
            )
            log.info("Downloading ocean loading data will be supported in the future")
            exit(1)

        self._resources["source"] = source

        return None

    def load_resources(self) -> None:

        with self._resources["source"].open("r") as f:

            content = f.readlines()

            for baseline in self.exp.baselines.values():

                _amp: np.ndarray | None = None
                _phs: np.ndarray | None = None
                for idx, line in enumerate(content):
                    line = line.strip()
                    if len(line) == 0 or line[0] == "$":
                        continue
                    if any([name in line for name in baseline.station.possible_names]):
                        idx += 1
                        while content[idx][0] == "$":
                            idx += 1
                        line = content[idx].strip()
                        _amp = (
                            np.array(
                                " ".join(content[idx : idx + 3]).split(), dtype=float
                            )
                            .reshape((3, 11))
                            .T
                        )
                        _phs = (
                            np.array(
                                " ".join(content[idx + 3 : idx + 6]).split(),
                                dtype=float,
                            )
                            .reshape((3, 11))
                            .T
                        )
                        break

                if _amp is None or _phs is None:
                    log.error(
                        f"Failed to load ocean loading data for {baseline.station}"
                    )
                    exit(1)
                self.resources[baseline.station.name] = (_amp, _phs)

        return None

    def calculate(self, station: "Station", epochs: "time.Time") -> None:

        return None


class SolidTide(Displacement):
    """Displacement due to solid Earth tides induced by the Sun and the Moon"""

    name = "SolidTide"
    etc = {}
    models = ["dehan_inel"]
    requires_spice = True

    def ensure_resources(self) -> None:

        log.debug(f"{self.name} displacement does not require external resources")

        return None

    def load_resources(self) -> None:

        log.debug(f"{self.name} displacement does not require external resources")

        return None

    def calculate(self, station: "Station", epochs: "time.Time") -> None:

        # Get epochs as ET
        et = (epochs.tdb - J2000.tdb).sec  # type: ignore

        # State vectors of station in ITRF
        sta = np.array(station.location(epochs).geocentric).T

        # Position vector of the Sun and Moon in GCRS
        sun = np.array(spice.spkpos("sun", et, "J2000", "CN", "earth")[0])
        moon = np.array(spice.spkpos("moon", et, "J2000", "CN", "earth")[0])

        # Calculate rotation matrix from J2000 to ITRF

        raise NotImplementedError(
            "Incomplete implementation: "
            "Next step is calculating rotation matrix from J2000 to ITRF"
        )

        return None
