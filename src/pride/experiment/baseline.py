from typing import TYPE_CHECKING
from ..logger import log
from astropy import time
import numpy as np
from .. import coordinates as coord
from scipy import interpolate

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
        setattr(self, "__eops", None)
        setattr(self, "__icrf2itrf", None)
        setattr(self, "__seu2itrf", None)
        setattr(self, "__lat", None)
        setattr(self, "__lon", None)

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

    # def update_time_stamps(self) -> None:

    #     _tstamps = time.Time(
    #         [observation.tstamps for observation in self.observations],
    #         scale="utc",
    #     ).sort()
    #     setattr(self, "__tstamps", _tstamps)

    #     return None

    def update_with_observations(self) -> None:
        """Update baseline object with data derived from observations"""

        log.debug(f"Updating {self.id} baseline with observations")

        # Merge time stamps from all the observations
        # NOTE: They inherit the location data from the observation
        tstamps: time.Time = time.Time(
            [observation.tstamps for observation in self.observations],
            scale="utc",
        ).sort()  # type: ignore

        # Retrieve geodetic coordinates of the station from time stamps
        if tstamps.location is None:
            log.error(
                f"Failed to update {self.id} baseline with observations: "
                "Location data is missing from observation time-stamps"
            )
            exit(1)
        geodetic = tstamps.location.to_geodetic("GRS80")
        lat = np.array(geodetic.lat.rad, dtype=float)
        lon = np.array(geodetic.lon.rad, dtype=float)

        # Calculate ICRF-ITRF and SEU-ITRF rotation matrices
        eops = self.exp.eops.at_epoch(tstamps, unit="arcsec")
        icrf2itrf = coord.icrf2itrf(eops, tstamps)
        seu2itrf = coord.seu2itrf(lat, lon)

        # Update observations with calculated data
        for observation in self.observations:

            # Get the index of the time stamps associated with the observation
            flags = np.sum(
                observation.tstamps[None, :] == tstamps[:, None],
                axis=1,
                dtype=bool,
            )

            setattr(observation, "__icrf2itrf", icrf2itrf[flags])
            setattr(observation, "__seu2itrf", seu2itrf[flags])

        # Update baseline with calculated data
        setattr(self, "__tstamps", tstamps)
        setattr(self, "__eops", eops)
        setattr(self, "__icrf2itrf", icrf2itrf)
        setattr(self, "__seu2itrf", seu2itrf)
        setattr(self, "__lat", lat)
        setattr(self, "__lon", lon)

        return None

    @property
    def eops(self) -> np.ndarray:

        if getattr(self, "__eops") is None:
            log.error(f"EOPs not found for {self}")
            exit(0)
        return getattr(self, "__eops")

    @property
    def icrf2itrf(self) -> np.ndarray:

        if getattr(self, "__icrf2itrf") is None:
            log.error(f"ICRF to ITRF transformation not found for {self}")
            exit(0)
        return getattr(self, "__icrf2itrf")

    @property
    def seu2itrf(self) -> np.ndarray:

        if getattr(self, "__seu2itrf") is None:
            log.error(f"SEU to ITRF transformation not found for {self}")
            exit(0)
        return getattr(self, "__seu2itrf")

    @property
    def lat(self) -> np.ndarray:

        if getattr(self, "__lat") is None:
            log.error(f"Latitude not found for {self}")
            exit(0)
        return getattr(self, "__lat")

    @property
    def lon(self) -> np.ndarray:

        if getattr(self, "__lon") is None:
            log.error(f"Longitude not found for {self}")
            exit(0)
        return getattr(self, "__lon")

    def update_station_with_geophysical_displacements(self) -> None:

        log.debug(
            f"Updating {self.station.name} station with geophysical "
            "displacements"
        )

        # Shared resources
        shared_resources = {
            "station_names": self.station.possible_names,
            "eops": self.eops,
            "icrf2itrf": self.icrf2itrf,
            "seu2itrf": self.seu2itrf,
            "lat": self.lat,
            "lon": self.lon,
            "xsta_itrf": self.station.location(self.tstamps),
        }

        # Update tectonic corrected position with displacements
        xsta_itrf = shared_resources["xsta_itrf"]
        for model in self.exp.displacement_models:
            resources = model.load_resources(self.tstamps, shared_resources)
            xsta_itrf += model.calculate(self.tstamps, resources)

        # Convert corrected position to ICRF
        xsta_icrf = (
            self.icrf2itrf.swapaxes(-1, -2) @ xsta_itrf[:, :, None]
        ).squeeze()

        # Generate interpolation polynomials
        # NOTE: I checked the difference between xsta and the interpolated
        # locations for GR035 and it is never bigger than 1e-9 m for any of the
        # stations. This is orders of magnitude smaller than any of the
        # displacements we are considering, so interpolating should be safe.
        _interp_location_itrf = {
            "x": interpolate.interp1d(
                self.tstamps.jd, xsta_itrf[:, 0], kind="cubic"
            ),
            "y": interpolate.interp1d(
                self.tstamps.jd, xsta_itrf[:, 1], kind="cubic"
            ),
            "z": interpolate.interp1d(
                self.tstamps.jd, xsta_itrf[:, 2], kind="cubic"
            ),
        }
        setattr(self, "_interp_location_itrf", _interp_location_itrf)

        _interp_location_icrf = {
            "x": interpolate.interp1d(
                self.tstamps.jd, xsta_icrf[:, 0], kind="cubic"
            ),
            "y": interpolate.interp1d(
                self.tstamps.jd, xsta_icrf[:, 1], kind="cubic"
            ),
            "z": interpolate.interp1d(
                self.tstamps.jd, xsta_icrf[:, 2], kind="cubic"
            ),
        }
        setattr(self.station, "_interp_location_icrf", _interp_location_icrf)

        # Update geophysical corrections flag for station
        self.station.has_geophysical_corrections = True

        return None
