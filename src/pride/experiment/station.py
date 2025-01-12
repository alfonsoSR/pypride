from typing import TYPE_CHECKING, Literal, Any
from .. import io
from ..logger import log
from astropy import time, coordinates
import numpy as np
from .. import coordinates as coord
from scipy import interpolate

if TYPE_CHECKING:
    from .experiment import Experiment


class Station:
    """VLBI station

    Attributes
    ----------
    name :
        Station name
    possible_names :
        Alternatives names for the station
    """

    __slots__ = (
        "name",
        "possible_names",
        "is_phase_center",
        "has_tectonic_correction",
        "has_geophysical_corrections",
        "exp",
        "_ref_epoch",
        "_ref_location",
        "_ref_velocity",
        "_interp_xsta_itrf",
        "_interp_xsta_icrf",
        "_interp_vsta_itrf",
        "_interp_vsta_icrf",
    )

    def __init__(self, name: str) -> None:

        # Reference and alternative names
        self.name = name
        self.possible_names = [name]
        alternative_names = io.load_catalog("station_names.yaml")
        if name in alternative_names:
            self.possible_names += alternative_names[name]

        # Phase center flag
        self.is_phase_center = False
        self.has_tectonic_correction = False
        self.has_geophysical_corrections = False

        # Optional attributes
        self.exp: "Experiment" = NotImplemented
        self._ref_epoch: time.Time = NotImplemented
        self._ref_location: np.ndarray = NotImplemented
        self._ref_velocity: np.ndarray = NotImplemented
        self._interp_xsta_itrf: interpolate.interp1d = NotImplemented
        self._interp_xsta_icrf: interpolate.interp1d = NotImplemented
        self._interp_vsta_itrf: interpolate.interp1d = NotImplemented
        self._interp_vsta_icrf: interpolate.interp1d = NotImplemented

        return None

    def __getattribute__(self, name: str) -> Any:

        val = super().__getattribute__(name)
        if val is NotImplemented:
            log.error(f"Attribute {name} not set for {self.name} station")
            exit(1)
        return val

    @staticmethod
    def from_experiment(name: str, experiment: "Experiment") -> "Station":

        station = Station(name)
        setup = experiment.setup
        station.exp = experiment

        # Check if station is the phase center
        if station.name == setup.general["phase_center"]:
            station.is_phase_center = True

            if station.name == "GEOCENTR":
                return station
            else:
                raise NotImplementedError(
                    "Using an arbitrary station as phase center is not "
                    "supported yet"
                )

        # Station coordinates at reference epoch
        with io.internal_file(
            setup.catalogues["station_positions"]
        ).open() as f:

            content = f.readlines()

            # Reference epoch
            ref_epoch_str: str | None = None
            for line in content:
                if "EPOCH" in line:
                    ref_epoch_str = line.split()[-1]
                    continue  # NOTE: Why not replace with break?
            if ref_epoch_str is None:
                log.error(
                    f"Failed to initialize {station.name} station: "
                    "Reference epoch not found in "
                    f"{setup.catalogues['station_positions']}"
                )
                exit(1)
            station._ref_epoch = time.Time.strptime(
                ref_epoch_str, "%Y.%m.%d", scale="utc"
            )

            # Station coordinates
            matching_position: str | None = None
            for line in content:
                line = line.strip()
                if len(line) == 0 or line[0] == "$":
                    continue
                if any([name in line for name in station.possible_names]):
                    matching_position = line
                    break
            if matching_position is None:
                log.error(
                    f"Failed to initialize {station.name} station: "
                    "Coordinates not found in "
                    f"{setup.catalogues['station_positions']}"
                )
                exit(1)
            station._ref_location = np.array(
                matching_position.split()[1:4], dtype=float
            )

        # Station velocity at reference epoch
        with io.internal_file(
            setup.catalogues["station_velocities"]
        ).open() as f:

            matching_velocity: str | None = None
            for line in f:
                line = line.strip()
                if len(line) == 0 or line[0] == "$":
                    continue
                if any([name in line for name in station.possible_names]):
                    matching_velocity = line
                    break
            if matching_velocity is None:
                log.error(
                    f"Failed to initialize {station.name} station: "
                    f"Velocity not found in {setup.catalogues['station_velocities']}"
                )
                exit(1)
            station._ref_velocity = (
                np.array(matching_velocity.split()[1:4], dtype=float) * 1e-3
            )

            station.has_tectonic_correction = True

        return station

    def tectonic_corrected_location(
        self, epoch: "time.Time"
    ) -> "coordinates.EarthLocation":
        """Station coordinates corrected for tectonic motion"""

        dt = (epoch.utc - self._ref_epoch.utc).to("year").value  # type: ignore
        coords = self._ref_location + (self._ref_velocity[:, None] * dt).T

        return coordinates.EarthLocation.from_geocentric(*coords.T, unit="m")

    def location(
        self, epoch: "time.Time", frame: Literal["itrf", "icrf"] = "itrf"
    ) -> np.ndarray:
        """Time-dependent station coordinates

        Returns cartesian coordinates of the station at a set of UTC epochs. The position includes the best available correction, which is determined based on the has_geophysical_corrections (G) flag.
        - G=True: Position is corrected for tectonic motion and all the geophysical displacements specified in the configuration file.
        - G=False: Position is corrected for tectonic motion only.

        :param epoch: UTC epochs at which to calculate the station coordinates
        :param frame: Reference frame for the coordinates: ITRF or GCRS
        :return: Station coordinates as (N, 3) array
        """

        if not self.has_tectonic_correction:
            raise NotImplementedError("Not supposed to happen")

        if not self.has_geophysical_corrections:

            out = np.array(self.tectonic_corrected_location(epoch).geocentric).T
            match frame:
                case "icrf":
                    eops = self.exp.eops.at_epoch(epoch, unit="arcsec")
                    return (
                        coord.itrf2icrf(eops, epoch) @ out[:, :, None]
                    ).squeeze()
                case "itrf":
                    return out
                case _:
                    log.error(
                        f"Failed to calculate {self.name} station coordinates: "
                        f"Invalid frame {frame}"
                    )

        # Get interpolation polynomials in chosen frame
        interp_location = getattr(self, f"_interp_xsta_{frame}")
        if interp_location is NotImplemented:
            log.error(
                f"Failed to calculate {self.name} station coordinates: "
                f"Interpolation polynomials not found for {frame} frame"
            )
            exit(1)

        # Ensure epoch is UTC
        if epoch.scale != "utc":
            log.warning(
                f"Converting epoch to UTC for {self.name} station coordinates"
            )
            epoch = epoch.utc  # type: ignore

        return interp_location(epoch.jd)

    def velocity(
        self, epoch: "time.Time", frame: Literal["itrf", "icrf"] = "itrf"
    ) -> np.ndarray:
        """Time-dependent station velocity

        Returns cartesian velocity of the station at a set of UTC epochs. The velocity is only available after the station has been updated with geophysical displacements.

        :param epoch: UTC epochs at which to calculate the station velocity
        :param frame: Reference frame for the velocity: ITRF or GCRS
        :return: Station velocity as (N, 3) array
        """

        if not self.has_tectonic_correction:
            raise NotImplementedError("Not supposed to happen")

        if not self.has_geophysical_corrections:
            log.error(
                f"Failed to calculate {self.name} station velocity: "
                "Geophysical displacements not applied"
            )
            exit(1)

        # Get interpolation polynomials in chosen frame
        interp_velocity = getattr(self, f"_interp_vsta_{frame}")
        if interp_velocity is NotImplemented:
            log.error(
                f"Failed to calculate {self.name} station velocity: "
                f"Interpolation polynomials not found for {frame} frame"
            )
            exit(1)

        # Ensure epoch is UTC
        if epoch.scale != "utc":
            log.warning(
                f"Converting epoch to UTC for {self.name} station velocity"
            )
            epoch = epoch.utc  # type: ignore

        return interp_velocity(epoch.jd)
