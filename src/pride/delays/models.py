from .core import Delay
from ..io import get_target_information, ESA_SPICE, NASA_SPICE, internal_file
from typing import TYPE_CHECKING, Any
from ..logger import log
import requests
from astropy import time
from ftplib import FTP_TLS
import unlzw3
import gzip
import numpy as np
from scipy import interpolate
from ..types import Antenna
from ..experiment.source import NearFieldSource, FarFieldSource

if TYPE_CHECKING:
    from pathlib import Path
    from ..experiment.station import Station
    from ..experiment.observation import Observation


class Geometric(Delay):
    """Geometric delay"""

    name = "Geometric"
    etc = {}
    requires_spice = True
    station_specific = False

    def ensure_resources(self) -> None:

        # Find kernels for target
        target = get_target_information(self.exp.setup.general["target"])
        kernel_path: "Path" = self.config["data"] / target["short_name"]
        metak_path: "Path" = kernel_path / "metak.tm"
        kernel_source = ESA_SPICE if target["api"] == "ESA" else NASA_SPICE
        kernel_source += f"/{target['names'][-1]}/kernels"

        # Create kernel directory if not present
        if not kernel_path.exists():
            kernel_path.mkdir(parents=True)
            log.info(f"Created kernel directory for {target['names'][-1]}")

        # Download metakernel if not present
        if not metak_path.exists():
            log.info(f"Downloading metakernel for {target['names'][-1]}")
            metak_found: bool = False
            response = requests.get(
                f"{kernel_source}/mk/{target['meta_kernel'].upper()}"
            )
            if response.ok:
                metak_found = True
                metak_content = response.content.decode("utf-8").splitlines()
                with metak_path.open("w") as f:
                    for line in metak_content:
                        if "PATH_VALUES" in line:
                            line = line.replace("..", str(kernel_path))
                        f.write(f"{line}\n")
            if not metak_found:
                log.error(
                    "Failed to initialize geometric delay: "
                    f"Metakernel not found for {target['names'][-1]}"
                )
                exit(1)

        # Read list of kernels from metakernel
        klist: list[str] = []
        with metak_path.open() as metak:
            content = iter([line.strip() for line in metak.readlines()])
            for line in content:
                if "KERNELS_TO_LOAD" in line:
                    line = next(content)
                    while ")" not in line:
                        if len(line) > 0:
                            klist.append(line.replace("$KERNELS/", "")[1:-1])
                        line = next(content)
                    if line.split(")")[0] != "":
                        klist.append(
                            line.split(")")[0].replace("$KERNELS/", "")[1:-1]
                        )
                    break

        # Ensure that all kernels are present
        for kernel in klist:
            if not (kernel_path / kernel).exists():
                log.info(f"Downloading {kernel_source}/{kernel}")
                response = requests.get(f"{kernel_source}/{kernel}")
                if response.ok:
                    (kernel_path / kernel).parent.mkdir(
                        parents=True, exist_ok=True
                    )
                    (kernel_path / kernel).write_bytes(response.content)
                else:
                    log.error(
                        "Failed to initialize geometric delay: "
                        f"Failed to download {kernel_source}/{kernel}"
                    )
                    exit(1)

        return None

    def load_resources(self) -> dict[str, Any]:
        return {}

    def calculate(self, obs: "Observation") -> Any:

        return NotImplemented


class Tropospheric(Delay):
    """Tropospheric delay

    # Available models
    ## Petrov
    - BRIEF DESCRIPTION OF THE MODEL AND SOURCE
    - DESCRIPTION OF REQUIRED RESOURCES AND KEYS

    ## Vienna Mapping Function 3 (VMF3)
    - BRIEF DESCRIPTION OF THE MODEL AND SOURCE
    - DESCRIPTION OF REQUIRED RESOURCES AND KEYS
    """

    name = "Tropospheric"
    etc = {
        "petrov_url": "http://pathdelay.net/spd/asc/geosfpit",
        "vienna_url": "https://vmf.geo.tuwien.ac.at/trop_products",
    }
    models: list[str] = ["petrov", "vienna"]
    requires_spice = False
    station_specific = True

    def ensure_resources(self) -> None:

        # Ensure resources for main tropospheric model
        if self.config["model"] not in self.models:
            log.error(
                "Failed to initialize tropospheric delay: "
                f"Invalid model {self.config['model']}"
            )
            exit(1)
        getattr(self, f"ensure_resources_{self.config['model']}")()

        # No further action if backup model is not enabled
        if not self.config["backup"]:
            return None

        # Ensure resources for backup model
        if self.config["backup_model"] not in self.models:
            log.error(
                "Failed to initialize tropospheric delay: "
                f"Invalid backup model {self.config['backup_model']}"
            )
            exit(1)
        getattr(self, f"ensure_resources_{self.config['backup_model']}")()

        return None

    def ensure_resources_petrov(self) -> None:

        # Initialize date from which to look for tropospheric data
        date: Any = time.Time(
            self.exp.initial_epoch.mjd // 1, format="mjd", scale="utc"  # type: ignore
        )
        step = time.TimeDelta(3 * 3600, format="sec")
        date += (self.exp.initial_epoch.datetime.hour // 3) * step  # type: ignore

        # Check if tropospheric data is available
        coverage: dict[tuple[time.Time, time.Time], Path] = {}
        while True:

            # Define path to tropospheric data file
            spd_file = self.config["data"] / (
                # spd_file = self.exp.setup.catalogues["tropospheric_data"] / (
                f"spd_geosfpit_{date.strftime('%Y%m%d')}_"
                f"{date.datetime.hour:02d}00.spd"
            )

            # Ensure parent directory exists
            if not spd_file.parent.exists():
                spd_file.parent.mkdir(parents=True, exist_ok=True)

            # Download file if not present
            if not spd_file.exists():
                log.info(f"Downloading {spd_file}")
                response = requests.get(
                    f"{self.etc['petrov_url']}/{spd_file.name}"
                )
                if response.ok:
                    spd_file.write_bytes(response.content)
                else:
                    raise FileNotFoundError(
                        "Failed to initialize tropospheric delay: "
                        f"Failed to download {self.etc['petrov_url']}/{spd_file.name}"
                    )

            # Add coverage to dictionary
            coverage[(date, date + step)] = spd_file

            # Update date
            if date > self.exp.final_epoch:
                break
            date += step

        # Add coverage to resources
        self.resources["coverage_petrov"] = coverage

        return None

    def ensure_resources_vienna(self) -> None:

        # Initialize date from which to look for tropospheric data
        date: Any = time.Time(
            self.exp.initial_epoch.mjd // 1, format="mjd", scale="utc"  # type: ignore
        )
        step = time.TimeDelta(6 * 3600, format="sec")
        date += (self.exp.initial_epoch.datetime.hour // 6) * step  # type: ignore

        # Define coverage dictionary
        site_coverage: dict[tuple[time.Time, time.Time], Path] = {}
        grid_coverage: dict[tuple[time.Time, time.Time], Path] = {}

        # Acquire tropospheric data
        while True:

            # Get year and day of year
            year = date.datetime.year
            doy = date.datetime.timetuple().tm_yday

            # Site-specific data
            site_file = self.config["data"] / f"{year:04d}{doy:03d}.v3gr_r"
            site_url = (
                self.etc["vienna_url"]
                + f"/VLBI/V3GR/V3GR_OP/daily/{year:04d}/{site_file.name}"
            )
            if not site_file.parent.exists():
                site_file.parent.mkdir(parents=True, exist_ok=True)
            if not site_file.exists():
                log.info(f"Downloading {site_file}")
                response = requests.get(site_url)
                if not response.ok:
                    raise FileNotFoundError(
                        "Failed to initialize tropospheric delay: "
                        f"Failed to download {site_url}"
                    )
                site_file.write_bytes(response.content)

            if site_file not in site_coverage.values():
                site_coverage[(date, date + step)] = site_file
            else:
                key = list(site_coverage.keys())[
                    list(site_coverage.values()).index(site_file)
                ]
                site_coverage.pop(key)
                site_coverage[(key[0], date + step)] = site_file

            # Grid data
            month = date.datetime.month
            day = date.datetime.day
            hour = date.datetime.hour

            grid_file = (
                self.config["data"]
                / f"V3GR_{year:04d}{month:02d}{day:02d}.H{hour:02d}"
            )
            grid_url = (
                f"{self.etc['vienna_url']}/GRID/1x1/V3GR/"
                f"V3GR_OP/{year:04d}/{grid_file.name}"
            )

            if not grid_file.parent.exists():
                grid_file.parent.mkdir(parents=True, exist_ok=True)
            if not grid_file.exists():
                log.info(f"Downloading {grid_file}")
                response = requests.get(grid_url)
                if not response.ok:
                    raise FileNotFoundError(
                        "Failed to initialize tropospheric delay: "
                        f"Failed to download {grid_url}"
                    )
                grid_file.write_bytes(response.content)

            grid_coverage[(date, date + step)] = grid_file

            # Update date
            if date > self.exp.final_epoch:
                break
            date += step

        # Add coverage to resources
        self.resources["coverage_vienna_site"] = site_coverage
        self.resources["coverage_vienna_grid"] = grid_coverage

    def load_resources(self) -> dict[str, tuple[str, Any]]:

        resources: dict[str, tuple[str, Any]] = {}

        for baseline in self.exp.baselines:

            # Check if resources are already available for station
            station = baseline.station
            if station.name in resources:
                continue

            # Attempt to load resources for main model
            success: bool = True
            try:
                resources_main = getattr(
                    self, f"load_resources_{self.config['model']}"
                )(station)
            except ValueError as e:
                if f"Station {station.name} not found" not in str(e):
                    raise e
                success = False

            # If successful, add resources to dictionary
            if success:
                resources[station.name] = (self.config["model"], resources_main)
                continue

            # Raise error if backup model is not enabled
            if not self.config["backup"]:
                log.error(
                    f"Failed to load resources for tropospheric delay: "
                    f"Station {station.name} not found in {self.config['model']} data"
                )
                exit(1)

            # Attempt to load resources for backup model
            log.warning(
                f"Using backup tropospheric model for {station.name} station"
            )
            resources_backup = getattr(
                self, f"load_resources_{self.config['backup_model']}"
            )(station)
            resources[station.name] = (
                self.config["backup_model"],
                resources_backup,
            )

        return resources

    def load_resources_petrov(self, station: "Station") -> dict[str, Any]:

        interp_dry: dict[time.Time, interpolate.CloughTocher2DInterpolator] = {}
        interp_wet: dict[time.Time, interpolate.CloughTocher2DInterpolator] = {}
        elevation_cutoff: dict[time.Time, np.ndarray] = {}

        for (t0, _), source in self.resources["coverage_petrov"].items():

            with source.open() as _f:

                f = _f.readlines()

                # Get station id
                station_id = ""
                for line in f:
                    if line[0] == "S" and np.any(
                        [name in line for name in station.possible_names]
                    ):
                        station_id = line.split()[1]
                        break
                if station_id == "":
                    raise ValueError(
                        "Failed to load tropospheric delay: "
                        f"Station {station.name} not found in {source}"
                    )

                # Get azimuth and elevation grids
                az_grid = {}
                el_grid = {}
                for line in f:
                    if line[0] == "A":
                        id, val = line.split()[1:]
                        az_grid[id] = float(val)
                    elif line[0] == "E":
                        id, val = line.split()[1:]
                        el_grid[id] = float(val)

                # Get tropospheric delay data
                data = {}
                for line in f:
                    if line[0] == "D" and line.split()[1] == station_id:
                        eid, aid, d1, d2 = line.split()[2:]
                        data[(el_grid[eid], az_grid[aid])] = (
                            float(d1.replace("D", "e")),
                            float(d2.replace("D", "e")),
                        )

            # Make grids and isolate dry and wet components
            spd_grid = np.array([list(key) for key in data.keys()])
            dry, wet = np.array(list(data.values())).T

            # Epoch in TAI
            epoch = time.Time(t0.datetime, scale="tai")

            # Interpolators
            interp_dry[epoch] = interpolate.CloughTocher2DInterpolator(
                spd_grid, dry
            )
            interp_wet[epoch] = interpolate.CloughTocher2DInterpolator(
                spd_grid, wet
            )
            elevation_cutoff[epoch] = np.min(spd_grid[:, 0])

        return {
            "dry": interp_dry,
            "wet": interp_wet,
            "elevation_cutoff": elevation_cutoff,
        }

    def load_resources_vienna(self, station: "Station") -> dict[str, Any]:

        site_present: bool = True
        data = []

        # Try to load site-specific data
        site_data = []
        for (t0, _), source in self.resources["coverage_vienna_site"].items():

            with source.open() as file:

                content: str = ""
                for line in file:
                    if np.any(
                        [name in line for name in station.possible_names]
                    ):
                        content = line
                        break

                if content == "":
                    site_present = False
                    break

            site_data.append(
                [float(x) for x in content.split()[1:6]]
                + [float(x) for x in content.split()[9:]]
            )

        # Load grid data if site-specific data is not available
        if site_present:
            data = np.array(site_data).T
        else:
            log.warning(
                f"Using gridded tropospheric data for {station.name} station"
            )
            grid_data: list[np.ndarray] = []

            for (t0, _), source in self.resources[
                "coverage_vienna_grid"
            ].items():

                data = np.loadtxt(source, skiprows=7).T
                lat = np.sort(np.unique(data[0]))
                lon = np.sort(np.unique(data[1]))
                grid_shape = (len(lat), len(lon))
                tmp = np.zeros((data.shape[0] - 1,))

                # Interpolate coefficients, delays and gradients for station
                tmp[0] = t0.mjd
                for idx in range(1, tmp.shape[0]):
                    (tmp[idx],) = interpolate.RectBivariateSpline(
                        lat, lon, data[idx + 1].reshape(grid_shape)
                    )(
                        station.location(t0).lat.deg,
                        station.location(t0).lon.deg,
                    )[
                        0
                    ]

                grid_data.append(tmp)

            data = np.array(grid_data).T

        # Generate interpolators
        interp_type: str = "linear" if len(data[0]) <= 3 else "cubic"
        return {
            "a_hydro": interpolate.interp1d(data[0], data[1], kind=interp_type),
            "a_wet": interpolate.interp1d(data[0], data[2], kind=interp_type),
            "d_hydro": interpolate.interp1d(data[0], data[3], kind=interp_type),
            "d_wet": interpolate.interp1d(data[0], data[4], kind=interp_type),
            "gn_hydro": interpolate.interp1d(
                data[0], data[5], kind=interp_type
            ),
            "ge_hydro": interpolate.interp1d(
                data[0], data[6], kind=interp_type
            ),
            "gn_wet": interpolate.interp1d(data[0], data[7], kind=interp_type),
            "ge_wet": interpolate.interp1d(data[0], data[8], kind=interp_type),
        }


class Ionospheric(Delay):
    """Ionospheric delay

    # Available models
    ## [MISSING NAME OF MODEL]
    - BRIEF DESCRIPTION OF THE MODEL AND SOURCE
    - DESCRIPTION OF REQUIRED RESOURCES AND KEYS
    """

    name = "Ionospheric"
    etc = {
        "url": "https://cddis.nasa.gov/archive/gps/products/ionex",
        "new_format_week": 2238,
        "gps_week_ref": time.Time("1980-01-06T00:00:00", scale="utc"),
        "model": "igs",
        "ftp_server": "gdc.cddis.eosdis.nasa.gov",
        "solution_type": "FIN",
    }
    requires_spice = False
    station_specific = True

    def ensure_resources(self) -> None:

        # Define range of dates to look for ionospheric data
        date: time.Time = time.Time(
            self.exp.initial_epoch.mjd // 1, format="mjd", scale="utc"  # type: ignore
        )
        date.format = "iso"
        step = time.TimeDelta(1.0, format="jd")

        coverage: dict[tuple[time.Time, time.Time], Path] = {}
        while True:

            # Get gps week from date
            gps_week = int((date - self.etc["gps_week_ref"]).to("week").value)
            year = date.datetime.year  # type: ignore
            doy = date.datetime.timetuple().tm_yday  # type: ignore

            # Get file name and url for ionospheric data file
            if gps_week < self.etc["new_format_week"]:
                ionex_zip = f"igsg{doy:03d}0.{str(year)[2:]}i.Z"
            else:
                ionex_zip = (
                    f"IGS0OPS{self.etc['solution_type']}_{year:04d}{doy:03d}"
                    "0000_01D_02H_GIM.INX.gz"
                )
            ionex_file = self.config["data"] / ionex_zip
            # ionex_file = self.exp.setup.catalogues["ionospheric_data"] / ionex_zip
            url = f"{self.etc['url']}/{year:4d}/{doy:03d}/{ionex_zip}"

            # Ensure parent directory exists
            if not ionex_file.parent.exists():
                ionex_file.parent.mkdir(parents=True, exist_ok=True)

            # Download file if not present
            if not ionex_file.with_suffix("").exists():

                if not ionex_file.exists():

                    log.info(f"Downloading {ionex_file.name}")

                    ftp = FTP_TLS(self.etc["ftp_server"])
                    ftp.login(user="anonymous", passwd="")
                    ftp.prot_p()
                    ftp.cwd(
                        "gps/products/ionex/" + "/".join(url.split("/")[-3:-1])
                    )
                    if not ionex_file.name in ftp.nlst():
                        raise FileNotFoundError(
                            "Failed to initialize ionospheric delay: "
                            f"Failed to download {url}"
                        )
                    ftp.retrbinary(
                        f"RETR {ionex_file.name}", ionex_file.open("wb").write
                    )

                # Uncompress file
                if ionex_file.suffix == ".Z":
                    ionex_file.with_suffix("").write_bytes(
                        unlzw3.unlzw(ionex_file.read_bytes())
                    )
                    ionex_file.unlink()
                elif ionex_file.suffix == ".gz":
                    with gzip.open(ionex_file, "rb") as f_in:
                        ionex_file.with_suffix("").write_bytes(f_in.read())
                    ionex_file.unlink()
                else:
                    raise ValueError(
                        "Failed to initialize ionospheric delay: "
                        "Invalid ionospheric data format"
                    )

            # Add coverage to dictionary
            coverage[(date, date + step)] = ionex_file.with_suffix("")

            # Update date
            if date > self.exp.final_epoch:
                break
            date += step

        # Add coverage to resources
        self.resources["coverage"] = coverage

        return None

    def load_resources(self) -> dict[str, Any]:

        # Generate TEC maps
        tec_epochs = time.Time([key[0] for key in self.resources["coverage"]])
        _tec_maps: list[np.ndarray] = []

        for source in self.resources["coverage"].values():

            with source.open() as f:

                content = iter([line.strip() for line in f])
                lat_grid: np.ndarray | None = None
                lon_grid: np.ndarray | None = None

                while True:

                    try:
                        line = next(content)
                    except StopIteration:
                        break

                    # Define latitude and longitude grids
                    if "LAT1 / LAT2 / DLAT" in line:
                        l0, l1, dl = np.array(line.split()[:3], dtype=float)
                        lat_grid = np.arange(l0, l1 + dl / 2, dl)
                    if "LON1 / LON2 / DLON" in line:
                        l0, l1, dl = np.array(line.split()[:3], dtype=float)
                        lon_grid = np.arange(l0, l1 + dl / 2, dl)

                    if "START OF TEC MAP" not in line:
                        continue
                    assert "START OF TEC MAP" in line
                    assert lat_grid is not None and lon_grid is not None
                    next(content)
                    next(content)

                    # Read TEC map
                    grid = np.zeros((len(lat_grid), len(lon_grid)))
                    for i, _ in enumerate(lat_grid):

                        grid[i] = np.array(
                            " ".join(
                                [next(content).strip() for _ in range(5)]
                            ).split(),
                            dtype=float,
                        )
                        line = next(content)

                    assert "END OF TEC MAP" in line
                    _tec_maps.append(grid)

        tec_maps = [
            interpolate.RegularGridInterpolator((lon_grid, lat_grid), grid.T)
            for grid in _tec_maps
        ]

        # Generate interpolators for the baselines
        resources: dict[str, Any] = {}
        for baseline in self.exp.baselines:

            coords = baseline.station.location(tec_epochs)
            data = [
                tec_map([lon, lat])[0]
                for tec_map, lon, lat in zip(tec_maps, coords.lon.deg, coords.lat.deg)  # type: ignore
            ]
            interp_type: str = "linear" if len(data) <= 3 else "cubic"
            resources[baseline.station.name] = interpolate.interp1d(
                tec_epochs.mjd, data, kind=interp_type
            )

        return resources


class ThermalDeformation(Delay):
    """Thermal deformation of antennas

    # Available models
    ## Nothnagel
    - Source:  Nothnagel (2009) https://doi.org/10.1007/s00190-008-0284-z

    Required resources:
    - Temperature at station location (Obtained from site-specific Vienna)
    - Antenna information: Focus type, mount type, foundation height and thermal expansion coefficient and reference temperature
    """

    name = "ThermalDeformation"
    etc = {
        "url": "https://vmf.geo.tuwien.ac.at/trop_products",
    }
    requires_spice = False
    station_specific = True

    def ensure_resources(self) -> None:

        # Initialize date from which to look for atmospheric data
        date: Any = time.Time(
            self.exp.initial_epoch.mjd // 1, format="mjd", scale="utc"  # type: ignore
        )
        step = time.TimeDelta(6 * 3600, format="sec")
        date += (self.exp.initial_epoch.datetime.hour // 6) * step  # type: ignore

        # Define coverage dictionary
        site_coverage: dict[tuple[time.Time, time.Time], Path] = {}

        # Acquire tropospheric data
        while True:

            # Get year and day of year
            year = date.datetime.year
            doy = date.datetime.timetuple().tm_yday

            # Site-specific data
            site_file = self.config["data"] / f"{year:04d}{doy:03d}.v3gr_r"
            site_url = (
                self.etc["url"]
                + f"/VLBI/V3GR/V3GR_OP/daily/{year:04d}/{site_file.name}"
            )
            if not site_file.parent.exists():
                site_file.parent.mkdir(parents=True, exist_ok=True)
            if not site_file.exists():
                log.info(f"Downloading {site_file}")
                response = requests.get(site_url)
                if not response.ok:
                    raise FileNotFoundError(
                        "Failed to initialize tropospheric delay: "
                        f"Failed to download {site_url}"
                    )
                site_file.write_bytes(response.content)

            if site_file not in site_coverage.values():
                site_coverage[(date, date + step)] = site_file
            else:
                key = list(site_coverage.keys())[
                    list(site_coverage.values()).index(site_file)
                ]
                site_coverage.pop(key)
                site_coverage[(key[0], date + step)] = site_file

            # Update date
            if date > self.exp.final_epoch:
                break
            date += step

        # Add coverage to resources
        self.resources["coverage"] = site_coverage

        return None

    def load_resources(self) -> dict[str, Any]:

        resources: dict[str, tuple["Antenna", Any]] = {}

        for baseline in self.exp.baselines:

            station = baseline.station
            if station in resources:
                continue

            # Generate antenna object from thermal deformation data file
            _antenna = Antenna(ivs_name=station.name)
            with internal_file(
                self.exp.setup.catalogues["antenna_parameters"]
            ).open() as f:

                matching_antenna: str | None = None
                for line in f:
                    line = line.strip()
                    if len(line) == 0 or line[0] == "#":
                        continue
                    if "ANTENNA_INFO" in line and any(
                        [x in line for x in station.possible_names]
                    ):
                        matching_antenna = line
                        break

                if matching_antenna is None:
                    log.warning(
                        f"Thermal deformation disabled for {station.name}: "
                        "Missing antenna parameters"
                    )
                    resources[station.name] = (_antenna, None)
                    continue
                else:
                    _antenna = Antenna.from_string(matching_antenna)

            # Load atmospheric data from site-specific Vienna files
            data = []
            for source in self.resources["coverage"].values():

                with source.open() as file:

                    content: str = ""
                    for line in file:
                        if np.any(
                            [name in line for name in station.possible_names]
                        ):
                            content = line
                            break

                    if content == "":
                        log.warning(
                            f"Thermal deformation disabled for {station.name}: "
                            "Missing atmospheric data"
                        )
                        break

                    _data = [float(x) for x in content.split()[1:]]
                    data.append([_data[0]] + _data[5:8])

            # Flag station if no data is available
            if len(data) == 0:
                resources[station.name] = (_antenna, None)
                continue

            data = np.array(data).T

            # Calculate humidity
            hum = self.humidity_model(data[2], data[3])

            interp_type: str = "linear" if len(data[0]) <= 3 else "cubic"
            _thermo = {
                "p": interpolate.interp1d(data[0], data[1], kind=interp_type),
                "TC": interpolate.interp1d(data[0], data[2], kind=interp_type),
                "hum": interpolate.interp1d(data[0], hum, kind=interp_type),
            }

            # Add station to resources
            resources[station.name] = (_antenna, _thermo)

        return resources

    @staticmethod
    def humidity_model(temp_c: np.ndarray, wvp: np.ndarray) -> np.ndarray:
        """Calculate relative humidity from temperature and water vapour pressure

        NOTE: Copy pasted from Dima. God knows where does this come from.
        """

        # Constants
        a = 10.79574
        c2k = 273.15
        b = 5.028
        c = 1.50475e-4
        d = 8.2969
        e = 0.42873e-3
        f = 4.76955
        g = 0.78614

        # Calculate saturation vapour pressure
        temp_k = temp_c + c2k
        ew = np.power(
            10.0,
            a * (1.0 - c2k / temp_k)
            - b * np.log10(temp_k / c2k)
            + c * (1 - np.power(10.0, d * (1.0 - temp_k / c2k)))
            + e * (np.power(10.0, f * (1.0 - temp_k / c2k)) - 1.0)
            + g,
        )

        # Calculate relative humidity
        return 100 * wvp / ew
