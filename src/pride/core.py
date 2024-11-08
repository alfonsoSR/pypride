from .io import Setup, VexContent, Vex, VEX_DATE_FORMAT, load_catalog, internal_file
from .logger import log
from typing import Any, TYPE_CHECKING, Generator
import numpy as np
from astropy import time, coordinates
from functools import reduce
import datetime
from pathlib import Path
from .delays import DELAY_MODELS
from contextlib import contextmanager
import spiceypy as spice
from .types import SourceType, Band, ObservationMode, Antenna

if TYPE_CHECKING:
    from .delays.core import Delay


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


class Scan:
    """Collection of epochs in which a source was observed

    :param id: Scan ID
    :param band: Band in which the source was observed
    :param start: Start time of the scan
    :param end: End time of the scan
    :param tstamps: Time stamps of the scan
    """

    def __init__(
        self,
        id: str,
        start: datetime.datetime,
        end: datetime.datetime,
        step: datetime.timedelta,
        band: Band,
        force_step: bool = False,
    ) -> None:
        """Initialize scan from start, end and step

        :param id: Scan ID
        :param start: Start time of the scan
        :param end: End time of the scan
        :param step: Time step between observations
        :param band: Band in which the source was observed
        :param force_step: Force the step size to be used
        """

        self.id = id
        self.band = band

        # Get time stamps
        duration = (end - start).total_seconds()
        _step = step.total_seconds()

        if np.modf(duration / _step)[0] < 1e-9:
            nobs = int(duration / _step) + 1
        else:
            if force_step:
                nobs = int(duration / _step) + 1
            else:
                log.warning(f"Bad step size for scan {self.id}: Using nearest integer")
                facs = self.factors(int(duration))
                pos = max(0, np.searchsorted(facs, _step) - 1)
                step = datetime.timedelta(seconds=facs[pos])
                nobs = int(duration / step.total_seconds()) + 1

        tstamps = [start + i * step for i in range(nobs)]
        if force_step:
            tstamps.append(end)

        self.tstamps = time.Time(tstamps, scale="utc")
        self.start: time.Time = self.tstamps[0]  # type: ignore
        assert isinstance(self.start, time.Time)
        self.end: time.Time = self.tstamps[-1]  # type: ignore
        assert isinstance(self.end, time.Time)

        return None

    @staticmethod
    def factors(n):
        # factorise a number
        facs = set(
            reduce(
                list.__add__,
                ([i, n // i] for i in range(1, int(n**0.5) + 1) if n % i == 0),
            )
        )
        return sorted(list(facs))


class Station:
    """VLBI station

    :param name: Station name
    :param possible_names: List of possible names for the station
    :param location: Station location corrected for tectonic motion
    :param antenna: Antenna parameters
    :param ocean_loading: Ocean loading parameters
    """

    def __init__(self, name: str, setup: Setup | None, epoch: time.Time | None) -> None:
        """Initialize station from setup and epoch

        :param name: Station name
        :param setup: Delay calculation setup
        :param epoch: Reference epoch for station location
        """

        # Reference and alternative names
        self.name = name
        self.possible_names = [name]
        alternative_names = load_catalog("station_names.yaml")
        if self.name in alternative_names:
            self.possible_names += alternative_names[self.name]

        # No further action for geocentric station
        if setup is None and epoch is None:
            return None
        assert setup is not None
        assert epoch is not None

        # Station location at reference epoch
        with internal_file(setup.catalogues["station_positions"]).open() as f:

            content = f.readlines()

            # Find reference epoch
            ref_epoch_str = ""
            for line in content:
                if "EPOCH" in line:
                    ref_epoch_str = line.split()[-1]
                    continue
            if ref_epoch_str == "":
                log.error(
                    f"Failed to generate Site object for {self.name}: Reference epoch "
                    f"not found in {setup.catalogues['station_positions']}"
                )
                exit(1)
            _ref_epoch = time.Time.strptime(ref_epoch_str, "%Y.%m.%d", scale="utc")

            # Find station position at reference epoch
            _matching_position: str | None = None
            for line in content:
                line = line.strip()
                if len(line) == 0 or line[0] == "$":
                    continue
                if any([name in line for name in self.possible_names]):
                    _matching_position = line
                    break
            if _matching_position is None:
                log.error(
                    f"Failed to generate Site object for {self.name}: "
                    f"Position not found in {setup.catalogues['station_positions']}"
                )
                exit(1)

            _position = np.array(_matching_position.split()[1:4], dtype=float)

        # Load station velocity vector at reference epoch
        with internal_file(setup.catalogues["station_velocities"]).open() as f:

            _matching_velocity: str | None = None
            for line in f:
                line = line.strip()
                if len(line) == 0 or line[0] == "$":
                    continue
                if self.name in line:
                    _matching_velocity = line
                    break
            if _matching_velocity is None:
                log.error(
                    f"Failed to generate {self.name} station: Velocity not found in "
                    f"{setup.catalogues['station_velocities']}"
                )
                exit(1)

            _velocity = np.array(_matching_velocity.split()[1:4], dtype=float) * 1e-3

        # Correct location for tectonic motion
        diff_epochs: float = (epoch - _ref_epoch).to("year").value  # type: ignore
        _position += diff_epochs * _velocity
        self.location = coordinates.EarthLocation.from_geocentric(*_position, unit="m")

        # # Load antenna parameters from catalog
        # with internal_file(setup.catalogues["antenna_parameters"]).open() as f:

        #     matching_antenna: str | None = None
        #     for line in f:
        #         line = line.strip()
        #         if len(line) == 0 or line[0] == "#":
        #             continue
        #         if "ANTENNA_INFO" in line and any(
        #             [x in line for x in self.possible_names]
        #         ):
        #             matching_antenna = line
        #             break

        #     if matching_antenna is None:
        #         if not setup.delays["ThermalDeformation"]["fallback"]:
        #             log.error(
        #                 f"Failed to generate {self.name} station: Antenna parameters not"
        #                 f" found in {setup.catalogues['antenna_parameters']}"
        #             )
        #             exit(1)
        #         else:
        #             log.warning(f"Using default antenna parameters for {self.name}")
        #             self.antenna = Antenna()
        #     else:
        #         self.antenna = Antenna.from_string(matching_antenna)

        # Ocean loading parameters
        with internal_file(setup.catalogues["ocean_loading"]).open() as f:

            content = f.readlines()
            ocean_loading: dict[str, np.ndarray] = {}

            for idx, line in enumerate(content):
                line = line.strip()
                if len(line) == 0 or line[0] == "$":
                    continue
                if any([name in line for name in self.possible_names]):
                    idx += 1
                    while content[idx][0] == "$":
                        idx += 1
                    line = content[idx].strip()
                    ocean_loading["amp"] = (
                        np.array(" ".join(content[idx : idx + 3]).split(), dtype=float)
                        .reshape((3, 11))
                        .T
                    )
                    ocean_loading["phs"] = (
                        np.array(
                            " ".join(content[idx + 3 : idx + 6]).split(), dtype=float
                        )
                        .reshape((3, 11))
                        .T
                    )
                    break
            if len(ocean_loading) == 0:
                log.error(
                    f"Failed to generate {self.name} station: Ocean loading parameters "
                    f"not found in {setup.catalogues['ocean_loading']}"
                )
                exit(1)

            self.ocean_loading: dict[str, np.ndarray] = ocean_loading

        return None

    @classmethod
    def copy(cls, obj: "Station") -> "Station":
        """Generate a copy of a station object"""

        station = Station(obj.name, None, None)
        station.location = obj.location.copy()
        station.ocean_loading = obj.ocean_loading.copy()

        return station

    def __repr__(self) -> str:
        return f"{self.name:10s}: {super().__repr__()}"


class Observation:
    """Observation

    A collection of scans of a source from a specific baseline

    :param baseline: Pair of stations forming the baseline [phase center + station]
    :param source: Observed source
    :param scans: List of scans
    """

    def __init__(self, baseline: tuple[Station, Station], source: Source) -> None:
        """Initialize observation from baseline and source

        :param baseline: Pair of stations forming the baseline [phase center + station]
        :param source: Observed source
        """

        self.baseline = baseline
        self.source = source
        self.scans: list[Scan] = []
        self._start: time.Time | None = None
        self._end: time.Time | None = None

        return None

    @property
    def empty(self) -> bool:
        return len(self.scans) == 0

    @property
    def start(self) -> time.Time:
        if self._start is None:
            raise ValueError("No scans in observation")
        return self._start

    @property
    def end(self) -> time.Time:
        if self._end is None:
            raise ValueError("No scans in observation")
        return self._end

    def add_scan(self, scan: Scan) -> None:

        self.scans.append(scan)

        if self._start is None or scan.start < self._start:
            self._start = scan.start
        if self._end is None or scan.end > self._end:
            self._end = scan.end

        return None

    def add_resources(self, delays: list["Delay"]) -> None:

        return None


class Experiment:
    """VLBI experiment

    :param name: Experiment name
    :param initial_epoch: Initial epoch
    :param final_epoch: Final epoch
    :param modes: Observation modes
    :param stations: Experiment stations
    :param sources: Experiment sources
    :param observations: List of observations
    :param setup: Delay calculation setup
    """

    def __init__(self, setup: str | Path) -> None:
        """Initialize experiment from VEX and configuration files

        :param setup: Path to setup file
        """

        # Parse VEX and configuration
        self.setup = Setup(setup) if isinstance(setup, str) else Setup(str(setup))
        vex = self.setup.general["vex"]
        _vex = Vex(vex) if isinstance(vex, str) else Vex(str(vex))

        # Experiment name
        self.name = _vex["GLOBAL"]["EXPER"]
        log.info(f"Initializing experiment object: {self.name}")

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

        # Constants for chosen version of ephemerides
        # NOTE: Current method to identify ephemerides version is not robust
        # self.constants = Constants.from_setup(self.setup)

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

        # Eliminate duplicated and ignored stations
        _ids = reversed(list(_experiment_stations.keys()))
        _names = reversed(list(_experiment_stations.values()))
        self.stations: dict[str, Station] = {
            k: Station(v, self.setup, self.initial_epoch)
            for k, v in reversed(dict(zip(_ids, _names)).items())
            if k not in self.setup.general["ignore_stations"]
        }

        # Define phase center
        if self.setup.general["phase_center"] not in _experiment_stations.values():
            if self.setup.general["phase_center"] == "GEOCENTR":
                self.phase_center = Station("GEOCENTR", None, None)
            else:
                raise ValueError(
                    f"Failed to generate experiment object: "
                    f"Invalid phase center {self.setup.general['phase_center']}"
                )
        else:
            self.phase_center = Station(
                self.setup.general["phase_center"], self.setup, self.initial_epoch
            )

        # NOTE: Not sure about logic when phase center is not geocenter
        if self.phase_center.name in self.stations:
            raise NotImplementedError(
                "Using one station as phase center is currently not supported: "
                "Use GEOCENTR instead"
            )

        # Load sources
        self.sources = {s: Source.from_vex(s, _vex) for s in _vex["SOURCE"]}

        # Initialize observations for all the baselines
        self.observations: list[Observation] = []
        for _source in self.sources.values():
            for _station in self.stations.values():
                self.observations.append(
                    Observation((self.phase_center, _station), _source)
                )

        # Add scans to observations
        for scan_id, scan_data in _vex["SCHED"].items():

            scan_stations = scan_data.getall("station")
            base_start = datetime.datetime.strptime(scan_data["start"], VEX_DATE_FORMAT)
            mode = self.modes[scan_data["mode"]]
            sources = scan_data.getall("source")

            for _station_data in scan_stations:

                _code, _dt_start, _dt_end = _station_data[:3]
                dt_start = datetime.timedelta(seconds=int(_dt_start.split()[0]))
                dt_end = datetime.timedelta(seconds=int(_dt_end.split()[0]))

                if _code not in self.stations:
                    raise KeyError(
                        f"Failed to generate scan {scan_id}: "
                        f"station {_code} not found in experiment"
                    )
                scan_station = self.stations[_code]

                scan_start = base_start + dt_start
                scan_end = scan_start + dt_end
                scan_duration = (scan_end - scan_start).total_seconds()

                _nobs = scan_duration / self.setup.general["delay_step"] + 1
                scan_step = datetime.timedelta(
                    seconds=1 if _nobs < 10 else self.setup.general["delay_step"]
                )

                scan = Scan(
                    scan_id,
                    scan_start,
                    scan_end,
                    scan_step,
                    mode.get_station_band(_code),
                )

                for observation in self.observations:
                    if observation.source.name not in sources:
                        continue
                    if observation.baseline[1].name != scan_station.name:
                        continue
                    observation.add_scan(scan)

            # NOTE: Original code adds GEOCENTR observations of near field sources to
            # calculate Doppler. Instead of this, I have temporarily restricted the
            # phase center to the center of the Earth, as that would allow me to
            # use the first station in the baseline for Doppler if the source type is
            # NearField. This is a temporary solution until I figure out the logic of
            # Doppler calculations.

        # Remove empty observations
        self.observations = [obs for obs in self.observations if not obs.empty]

        # Initialize list of delays
        self.delays: list["Delay"] = []
        self.requires_spice: bool = False
        for key, delay_config in self.setup.delays.items():
            if delay_config["calculate"]:
                if key not in DELAY_MODELS:
                    log.error(
                        "Failed to initialize delay model: "
                        f"{key} delay is not implemented"
                    )
                    exit(1)
                self.delays.append(DELAY_MODELS[key](self))
                if DELAY_MODELS[key].requires_spice:
                    self.requires_spice = True

        # Update experiment with resources
        self.resources: dict[str, Any] = {
            delay.name: delay.load_resources() for delay in self.delays
        }

        return None

    @contextmanager
    def spice_kernels(self) -> Generator:
        """Context manager to load SPICE kernels"""

        try:
            if self.requires_spice:
                log.info("Loading SPICE kernels")
                metak = str(
                    self.setup.resources["ephemerides"]
                    / self.setup.general["target"]
                    / "metak.tm"
                )
                spice.furnsh(metak)

            yield None
        finally:
            if self.requires_spice:
                log.info("Unloading SPICE kernels")
                spice.kclear()

        return None

    def calculate_delays(self):
        """Calculate delays"""

        output = {}

        for observation in self.observations:

            output[observation] = {}

            if observation.baseline[1].name == "CEDUNA":
                continue

            for scan in observation.scans:

                output[observation][scan] = [
                    delay.calculate(observation, scan.tstamps) for delay in self.delays
                ]

            return output

        return None
