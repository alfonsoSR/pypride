from pathlib import Path
from ..io import Setup, Vex, VexContent, VEX_DATE_FORMAT, load_catalog, internal_file
from ..logger import log
from astropy import time, coordinates
from typing import Any, TYPE_CHECKING
from ..types import ObservationMode, SourceType, Band
import numpy as np
import datetime
from .. import utils
from ..delays import DELAY_MODELS

if TYPE_CHECKING:
    from ..delays.core import Delay


class Station:
    """VLBI station

    :param name: Station name
    :param possible_names: List of possible names for the station
    :param ref_epoch: Reference epoch at which the coordinates and velocity of the station are known
    :param ref_location: Coordinates of the station at reference epoch [ITRS]
    :param ref_velocity: Velocity of the station at reference epoch [ITRS]
    """

    def __init__(self, name: str) -> None:
        """Initialize empty station

        :param name: Station name
        """

        # Reference and alternative names
        self.name = name
        self.possible_names = [name]
        alternative_names = load_catalog("station_names.yaml")
        if self.name in alternative_names:
            self.possible_names += alternative_names[self.name]

        # Phase center flag
        self.is_phase_center: bool = False

        # Optional attributes
        setattr(self, "__ref_epoch", None)
        setattr(self, "__ref_location", None)
        setattr(self, "__ref_velocity", None)

        return None

    @property
    def ref_epoch(self) -> time.Time:
        if getattr(self, "__ref_epoch") is None:
            log.error(f"Reference epoch for {self.name} station not set")
            exit(1)
        return getattr(self, "__ref_epoch")

    @property
    def ref_location(self) -> np.ndarray:
        if getattr(self, "__ref_location") is None:
            log.error(f"Reference location for {self.name} station not set")
            exit(1)
        return getattr(self, "__ref_location")

    @property
    def ref_velocity(self) -> np.ndarray:
        if getattr(self, "__ref_velocity") is None:
            log.error(f"Reference velocity for {self.name} station not set")
            exit(1)
        return getattr(self, "__ref_velocity")

    @staticmethod
    def from_setup(name: str, setup: Setup) -> "Station":
        """Initialize station from configuration file

        :param name: Station name
        :param setup: Setup for delay calculation
        """

        # Initialize empty station
        station = Station(name)

        # Check if station is phase center
        if station.name == setup.general["phase_center"]:
            station.is_phase_center = True
            if station.name == "GEOCENTR":
                return station
            else:
                log.error(
                    "Using an arbitrary station as phase center "
                    "is currently not supported"
                )
                exit(0)

        # Station coordinates at reference epoch
        with internal_file(setup.catalogues["station_positions"]).open() as f:

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
                "__ref_epoch",
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
                    f"Coordinates not found in {setup.catalogues['station_positions']}"
                )
                exit(1)
            setattr(
                station,
                "__ref_location",
                np.array(matching_position.split()[1:4], dtype=float),
            )

        # Station velocity at reference epoch
        with internal_file(setup.catalogues["station_velocities"]).open() as f:

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
                "__ref_velocity",
                np.array(matching_velocity.split()[1:4], dtype=float) * 1e-3,
            )

        return station

    def location(self, epoch: time.Time) -> coordinates.EarthLocation:
        """Station coordinates at a given epoch

        Obtained correcting the reference coordinates for tectonic motion

        :param epoch: Epoch at which the coordinates are required
        """

        dt = (epoch - self.ref_epoch).to("year").value  # type: ignore
        corrected_position = self.ref_location + (self.ref_velocity[:, None] * dt).T
        return coordinates.EarthLocation(*corrected_position.T, unit="m")

    def __repr__(self) -> str:
        return f"{self.name:10s}: {super().__repr__()}"


class Source:
    """Source

    :param name: Source name
    :param source_type: Source type
    :param ra: Right ascension [rad]
    :param dec: Declination [rad]
    """

    def __init__(
        self, name: str, source_type: SourceType, ra: float, dec: float
    ) -> None:

        self.name = name
        self.type = source_type
        self.coords = coordinates.SkyCoord(ra, dec, frame="icrs", unit="rad")

        return None

    @property
    def K_s(self) -> np.ndarray:
        """Unit vector in the direction of the source"""

        ra: float = self.coords.ra.rad  # type: ignore
        dec: float = self.coords.dec.rad  # type: ignore

        return np.array(
            [
                np.cos(ra) * np.cos(dec),
                np.sin(ra) * np.cos(dec),
                np.sin(dec),
            ]
        )

    @staticmethod
    def __find_type(source_info: VexContent, _source_name: str) -> SourceType:

        if "source_type" not in source_info:
            raise ValueError(
                f"Failed to generate source object for {_source_name}: "
                "Missing type information in VEX file."
            )

        source_type = source_info["source_type"]

        if source_type == "calibrator":
            return SourceType.FarField
        elif source_type == "target":
            return SourceType.NearField
        else:
            raise TypeError(
                f"Failed to generate source object for {_source_name}: "
                f"Invalid type {source_type}"
            )

    @staticmethod
    def from_vex(source_name: str, vex: VexContent) -> "Source":

        if source_name not in vex["SOURCE"]:
            raise ValueError(f"Source {source_name} not found in VEX file")

        source_info = vex["SOURCE"][source_name]
        if source_info["ref_coord_frame"] != "J2000":
            raise NotImplementedError(
                f"Failed to generate source object for {source_name}: "
                f"Coordinate frame {source_info['ref_coord_frame']} not supported"
            )
        coords = coordinates.SkyCoord(
            source_info["ra"], source_info["dec"], frame="icrs"
        )

        return Source(
            source_name,
            Source.__find_type(source_info, source_name),
            coords.ra.rad,  # type: ignore
            coords.dec.rad,  # type: ignore
        )


class Observation:
    """Observation of a source

    The time stamps are stored as an astropy Time object including information about the coordinates of the station for which the observation was performed. The coordinates of the station are obtained by correcting the location at a reference epoch (loaded from data file) for tectonic motion.

    :param source: Source object representing the target
    :param band: Frequency band in which the source was detected
    :param tstamps: Collection of epochs in which the source was detected
    """

    def __init__(
        self,
        station: "Station",
        source: "Source",
        band: "Band",
        tstamps: list[datetime.datetime],
    ) -> None:
        """Initialize observation

        :param station: Station of the baseline from which the observation was performed
        :param source: Source object representing the target
        :param band: Frequency band in which the source was detected
        :param tstamps: List of UTC epochs in which the source was detected
        """

        self.source = source
        self.band = band
        _tstamps = time.Time(sorted(tstamps), scale="utc")
        self.tstamps = time.Time(
            sorted(tstamps), scale="utc", location=station.location(_tstamps)
        )

        return None


class Baseline:
    """VLBI baseline

    A baseline is a pair of stations that have observations of different sources associated with them

    :param center: Station object representing the phase center
    :param station: Station object representing the other station in the baseline
    :param observations: List of observations associated with the baseline
    :param nobs: Number of observations associated with the baseline
    """

    def __init__(self, center: "Station", station: "Station") -> None:
        """Initialize baseline

        :param center: Station object representing the phase center
        :param station: Station object representing the other station in the baseline
        """

        self.center = center
        self.station = station
        self.observations: list["Observation"] = []

        return None

    @property
    def nobs(self) -> int:
        return len(self.observations)

    def add_observation(self, observation: "Observation") -> None:
        """Add observation to baseline"""

        self.observations.append(observation)

        return None

    def __str__(self) -> str:
        return f"{self.center.name}-{self.station.name}"


class Experiment:
    """VLBI experiment

    :param setup: Setup for calculation of delays
    :param name: Experiment name
    :param initial_epoch: Initial epoch of the experiment
    :param final_epoch: Final epoch of the experiment
    :param modes: Observation modes [ADD BETTER DESCRIPTION]
    :param phase_center: Station object for the phase center of the experiment
    :param baselines: List of baselines in the experiment
    """

    def __init__(self, setup: str | Path) -> None:
        """Initialize experiment from configuration file

        :param setup: Path to configuration file
        """

        # Parse VEX and configuration
        self.setup = Setup(str(setup))
        _vex = Vex(str(self.setup.general["vex"]))

        # Experiment name
        self.name = _vex["GLOBAL"]["EXPER"]
        log.info(f"Initializing {self.name} experiment")

        # Initial and final epochs
        self.initial_epoch = time.Time.strptime(
            _vex["EXPER"][self.name]["exper_nominal_start"],
            VEX_DATE_FORMAT,
            scale="utc",
        )
        self.final_epoch = time.Time.strptime(
            _vex["EXPER"][self.name]["exper_nominal_stop"],
            VEX_DATE_FORMAT,
            scale="utc",
        )

        # Observation modes
        self.modes: dict[str, ObservationMode] = {
            mode: ObservationMode(mode, _vex["MODE"][mode].getall("FREQ"), _vex["FREQ"])
            for mode in _vex["MODE"]
        }

        # Sources
        self.sources: dict[str, "Source"] = {
            name: Source.from_vex(name, _vex) for name in _vex["SOURCE"]
        }

        # Load experiment stations
        _experiment_stations = {
            sta: _vex["STATION"][sta]["ANTENNA"] for sta in _vex["STATION"]
        }

        # Load station catalog
        _station_catalog = load_catalog("station_names.yaml")

        # Ensure that all the stations have valid names
        for id, exp_sta in _experiment_stations.items():
            for sta_name, sta_alternatives in _station_catalog.items():
                if exp_sta in sta_alternatives:
                    _experiment_stations[id] = sta_name
                    break

        # Define phase center
        self.phase_center = Station.from_setup(
            self.setup.general["phase_center"], self.setup
        )

        # Define baselines
        _ids = reversed(list(_experiment_stations.keys()))
        _names = reversed(list(_experiment_stations.values()))
        _baselines: dict[str, "Baseline"] = {
            k: Baseline(self.phase_center, Station.from_setup(v, self.setup))
            for k, v in reversed(dict(zip(_ids, _names)).items())
            if k not in self.setup.general["ignore_stations"]
        }
        self.baselines = list(_baselines.values())

        # Add observations to baselines
        observation_bands: dict[str, dict[str, "Band"]] = {}
        observation_tstamps: dict[str, dict[str, list[datetime.datetime]]] = {}
        for scan_id, scan_data in _vex["SCHED"].items():

            # Metadata
            scan_stations = scan_data.getall("station")
            base_start = datetime.datetime.strptime(scan_data["start"], VEX_DATE_FORMAT)
            mode = self.modes[scan_data["mode"]]
            if len(scan_data.getall("source")) > 1:
                raise NotImplementedError(f"Scan {scan_id} has multiple sources")
            source = scan_data.getall("source")[0]

            for station_data in scan_stations:

                # Retrieve station code and time window
                _code, _dt_start, _dt_end = station_data[:3]

                # Get baseline
                if _code not in _baselines:
                    log.error(
                        f"Failed to initialize {self.name} experiment: "
                        f"None of the baselines contains a station with code {_code}"
                    )
                    exit(1)

                # Beginning, end and duration of scan
                dt_start = datetime.timedelta(seconds=int(_dt_start.split()[0]))
                dt_end = datetime.timedelta(seconds=int(_dt_end.split()[0]))
                scan_start = base_start + dt_start
                scan_end = base_start + dt_end
                scan_duration = (dt_end - dt_start).total_seconds()

                # Define step size based on configuration file
                nobs = scan_duration / self.setup.general["scan_step"] + 1
                scan_step = 1 if nobs < 10 else self.setup.general["scan_step"]

                # Calculate timestamps
                if np.modf(scan_duration / scan_step)[0] < 1e-9:
                    nobs = int(scan_duration / scan_step) + 1
                else:
                    if self.setup.general["force_scan_step"]:
                        nobs = int(scan_duration / scan_step) + 1
                    else:
                        log.debug(
                            f"Adjusting discretization step for scan {scan_id} "
                            f"Duration {scan_duration} s, step {scan_step} s"
                        )
                        facs = utils.factors(int(scan_duration))
                        pos = max(0, int(np.searchsorted(facs, scan_step)) - 1)
                        scan_step = facs[pos]
                        nobs = int(scan_duration / scan_step) + 1
                scan_step = datetime.timedelta(seconds=scan_step)
                scan_tstamps = [scan_start + i * scan_step for i in range(nobs)]
                if self.setup.general["force_scan_step"]:
                    scan_tstamps.append(scan_end)

                # Update observation data
                if _code not in observation_bands:
                    observation_bands[_code] = {}
                    observation_tstamps[_code] = {}
                else:
                    band = mode.get_station_band(_code)
                    if source not in observation_bands[_code]:
                        observation_bands[_code][source] = band
                        observation_tstamps[_code][source] = scan_tstamps
                    else:
                        assert observation_bands[_code][source] == band
                        observation_tstamps[_code][source] += scan_tstamps

        # Update baselines with observations
        log.info("Loading observations")
        for baseline_id in observation_bands:
            for source_id in observation_bands[baseline_id]:
                _baselines[baseline_id].add_observation(
                    Observation(
                        _baselines[baseline_id].station,
                        self.sources[source_id],
                        observation_bands[baseline_id][source_id],
                        observation_tstamps[baseline_id][source_id],
                    )
                )

        # Add delays
        log.info("Setting up delay models")
        self.delays: list["Delay"] = []
        self.requires_spice: bool = False
        for delay_id, delay_config in self.setup.delays.items():
            if delay_config["calculate"]:
                if delay_id not in DELAY_MODELS:
                    log.error(f"Failed to initialize {delay_id} delay: Model not found")
                    exit(1)
                self.delays.append(DELAY_MODELS[delay_id](self))
                if DELAY_MODELS[delay_id].requires_spice:
                    self.requires_spice = True

        # Load resources
        self.resources: dict[str, Any] = {
            delay.name: delay.load_resources() for delay in self.delays
        }

        return None
