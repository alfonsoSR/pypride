from typing import TYPE_CHECKING, Generator
from pathlib import Path
from contextlib import contextmanager
import spiceypy as spice
from ..io import Setup, Vex, load_catalog
from ..logger import log
from astropy import time
import datetime
import numpy as np
from .. import utils
from ..types import ObservationMode, Band
from ..coordinates import EOP
from ..displacements import DISPLACEMENT_MODELS
from ..delays import DELAY_MODELS
from .source import Source
from .station import Station
from .baseline import Baseline
from .observation import Observation

if TYPE_CHECKING:
    from ..displacements.core import Displacement
    from ..delays.core import Delay


class Experiment:
    """VLBI experiment"""

    def __init__(self, setup: str | Path) -> None:

        # Parse VEX and configuration files
        self.setup = Setup(str(setup))
        self.vex = Vex(str(self.setup.general["vex"]))

        # Experiment name
        self.name = self.vex["GLOBAL"]["EXPER"]
        log.info(f"Initializing {self.name} experiment")

        # Initial and final epochs
        self.initial_epoch = time.Time.strptime(
            self.vex["EXPER"][self.name]["exper_nominal_start"],
            self.setup.internal["vex_date_format"],
            scale="utc",
        )
        self.final_epoch = time.Time.strptime(
            self.vex["EXPER"][self.name]["exper_nominal_stop"],
            self.setup.internal["vex_date_format"],
            scale="utc",
        )

        # Observation modes
        self.modes: dict[str, "ObservationMode"] = {
            mode: ObservationMode(
                mode, self.vex["MODE"][mode].getall("FREQ"), self.vex["FREQ"]
            )
            for mode in self.vex["MODE"]
        }

        # EOPs: For transformations between ITRF and ICRF
        self.eops = EOP.from_experiment(self)

        # Load sources
        self.sources: dict[str, "Source"] = {
            name: Source.from_experiment(name, self)
            for name in self.vex["SOURCE"]
        }

        # Define phase center
        self.phase_center = Station.from_experiment(
            self.setup.general["phase_center"], self
        )

        # Initialize delay and displacement models
        self.requires_spice = False
        self.displacement_models = self.initialize_displacement_models()
        self.delay_models = self.initialize_delay_models()

        # Initialize baselines
        self.baselines = self.initialize_baselines()

        return None

    def initialize_displacement_models(self) -> list["Displacement"]:

        log.info("Setting up geophysical displacements")

        displacement_models: list["Displacement"] = []
        for displacement, calculate in self.setup.displacements.items():
            if calculate:
                if displacement not in DISPLACEMENT_MODELS:
                    log.error(
                        f"Failed to initialize {displacement} displacement: "
                        "Model not found"
                    )
                    exit(1)
                displacement_models.append(
                    DISPLACEMENT_MODELS[displacement](self)
                )
                if DISPLACEMENT_MODELS[displacement].requires_spice:
                    self.requires_spice = True

        return displacement_models

    def initialize_delay_models(self) -> list["Delay"]:

        log.info("Setting up delay models...")

        _delay_models: list["Delay"] = []
        for delay_id, delay_config in self.setup.delays.items():
            if delay_config["calculate"]:
                if delay_id not in DELAY_MODELS:
                    log.error(
                        f"Failed to initialize {delay_id} delay: Model not found"
                    )
                    exit(1)
                _delay_models.append(DELAY_MODELS[delay_id](self))
                if DELAY_MODELS[delay_id].requires_spice:
                    self.requires_spice = True

        # # Load resources for delay models
        # raise NotImplementedError(
        #     "Delay objects should update themselves with resources, not load them into the experiment"
        # )
        # for delay in _delay_models:
        #     delay.load_resources()

        # Save as private property
        # self.__delays = _delay_models

        return _delay_models

    def initialize_baselines(self) -> list["Baseline"]:

        log.info("Setting up baselines")

        # Load experiment stations
        _experiment_stations = {
            sta: self.vex["STATION"][sta]["ANTENNA"]
            for sta in self.vex["STATION"]
        }

        # Load station catalog
        _station_catalog = load_catalog("station_names.yaml")

        # Ensure that all the stations have valid names
        for id, exp_sta in _experiment_stations.items():
            for sta_name, sta_alternatives in _station_catalog.items():
                if exp_sta in sta_alternatives:
                    _experiment_stations[id] = sta_name
                    break

        # Define baselines
        _ids = reversed(list(_experiment_stations.keys()))
        _names = reversed(list(_experiment_stations.values()))
        _baselines: dict[str, "Baseline"] = {
            k: Baseline(self.phase_center, Station.from_experiment(v, self))
            for k, v in reversed(dict(zip(_ids, _names)).items())
            if k not in self.setup.general["ignore_stations"]
        }

        # Add observations to baselines
        observation_bands: dict[str, dict[str, "Band"]] = {}
        observation_tstamps: dict[str, dict[str, list[datetime.datetime]]] = {}
        for scan_id, scan_data in self.vex["SCHED"].items():

            # Metadata
            scan_stations = scan_data.getall("station")
            base_start = datetime.datetime.strptime(
                scan_data["start"], self.setup.internal["vex_date_format"]
            )
            mode = self.modes[scan_data["mode"]]
            if len(scan_data.getall("source")) > 1:
                raise NotImplementedError(
                    f"Scan {scan_id} has multiple sources"
                )
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

            # Update baselines with observations
            for source_id in observation_bands[baseline_id]:
                _baselines[baseline_id].add_observation(
                    Observation(
                        _baselines[baseline_id],
                        self.sources[source_id],
                        observation_bands[baseline_id][source_id],
                        observation_tstamps[baseline_id][source_id],
                    )
                )

            # Group observation epochs of the baseline
            _baselines[baseline_id].update_with_observations()

        return list(_baselines.values())

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
