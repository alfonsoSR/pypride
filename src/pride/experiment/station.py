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

        # Private attributes
        setattr(self, "_exp", None)
        setattr(self, "_ref_epoch", None)
        setattr(self, "_ref_location", None)
        setattr(self, "_ref_velocity", None)
        setattr(self, "_interp_location_itrf", None)
        setattr(self, "_interp_location_icrf", None)

        return None

    @property
    def exp(self) -> "Experiment":

        if getattr(self, "_exp") is None:
            log.error(f"Experiment not set for {self.name} station")
            exit(1)
        return getattr(self, "_exp")

    @staticmethod
    def from_experiment(name: str, experiment: "Experiment") -> "Station":

        station = Station(name)
        setup = experiment.setup
        setattr(station, "_exp", experiment)

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
            setattr(
                station,
                "_ref_epoch",
                time.Time.strptime(ref_epoch_str, "%Y.%m.%d", scale="utc"),
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
            setattr(
                station,
                "_ref_location",
                np.array(matching_position.split()[1:4], dtype=float),
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
            setattr(
                station,
                "_ref_velocity",
                np.array(matching_velocity.split()[1:4], dtype=float) * 1e-3,
            )
            station.has_tectonic_correction = True

        return station

    def tectonic_corrected_location(
        self, epoch: "time.Time"
    ) -> "coordinates.EarthLocation":
        """Station coordinates corrected for tectonic motion"""

        dt = (epoch.utc - getattr(self, "_ref_epoch").utc).to("year").value
        coords = (
            getattr(self, "_ref_location")
            + (getattr(self, "_ref_velocity")[:, None] * dt).T
        )
        return coordinates.EarthLocation.from_geocentric(*coords.T, unit="m")

    def add_geophysical_displacements(self, epochs: "time.Time") -> None:

        log.debug(
            f"Updating {self.name} station with geophysical displacements"
        )

        # Geodetic coordinates of the station
        geodetic = self.tectonic_corrected_location(epochs).to_geodetic("GRS80")
        lat = np.array(geodetic.lat.rad, dtype=float)
        lon = np.array(geodetic.lon.rad, dtype=float)

        # Shared resources between displacements
        eops = self.exp.eops.at_epoch(epochs, unit="arcsec")
        shared_resources = {
            "station_names": self.possible_names,
            "eops": eops,
            "seu2itrf": coord.seu2itrf(lat, lon),
            "icrf2itrf": coord.icrf2itrf(eops, epochs),
            "lat": lat,
            "lon": lon,
            "xsta_itrf": self.location(epochs),
        }

        # Load resources for each displacement model
        resources: dict[str, Any] = {
            model.name: model.load_resources(epochs, shared_resources)
            for model in self.exp.displacement_models
        }

        # Correct station coordinates at observation epochs
        xsta_itrf = shared_resources["xsta_itrf"]
        for model in self.exp.displacement_models:
            xsta_itrf += model.calculate(epochs, resources[model.name])

        # Convert coordinates to geocentric ICRF
        xsta_icrf = (
            shared_resources["icrf2itrf"].swapaxes(-1, -2)
            @ xsta_itrf[:, :, None]
        ).squeeze()

        # Generate interpolation polymonials
        # NOTE: I checked the difference between xsta and the interpolated
        # locations for GR035 and it is never bigger than 1e-9 m for any of the
        # stations. This is orders of magnitude smaller than any of the
        # displacements we are considering, so interpolating should be safe.
        _interp_location_itrf = {
            "x": interpolate.interp1d(epochs.jd, xsta_itrf[:, 0], kind="cubic"),
            "y": interpolate.interp1d(epochs.jd, xsta_itrf[:, 1], kind="cubic"),
            "z": interpolate.interp1d(epochs.jd, xsta_itrf[:, 2], kind="cubic"),
        }
        setattr(self, "_interp_location_itrf", _interp_location_itrf)

        _interp_location_icrf = {
            "x": interpolate.interp1d(epochs.jd, xsta_icrf[:, 0], kind="cubic"),
            "y": interpolate.interp1d(epochs.jd, xsta_icrf[:, 1], kind="cubic"),
            "z": interpolate.interp1d(epochs.jd, xsta_icrf[:, 2], kind="cubic"),
        }
        setattr(self, "_interp_location_icrf", _interp_location_icrf)

        # Update geophysical corrections flag
        self.has_geophysical_corrections = True

        return None

    def location(
        self, epoch: "time.Time", frame: Literal["itrf", "gcrs"] = "itrf"
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
                case "gcrs":
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
        interp_location = getattr(self, f"_interp_location_{frame}")
        if interp_location is None:
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

        return np.array(
            [
                interp_location["x"](epoch.jd),
                interp_location["y"](epoch.jd),
                interp_location["z"](epoch.jd),
            ]
        ).T
