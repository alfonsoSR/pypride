import yaml
from pathlib import Path
from typing import Any
from .resources import internal_file, load_catalog


class Setup:
    """Delay calculation setup

    :param config_file: Path to configuration file
    """

    def __init__(self, config_file: str) -> None:

        with open(config_file) as f:
            config = yaml.safe_load(f)

        # Paths to internal and external files
        external = Path().home() / ".pride"

        # Internal configuration
        internal_config = load_catalog("config.yaml")
        self.catalogues: dict[str, Any] = internal_config["Catalogues"]

        # User configuration
        self.general: dict[str, Any] = config["Experiment"]

        self.resources: dict[str, Path] = config["Resources"]
        for key, val in self.resources.items():
            self.resources[key] = external / val

        self.displacements: dict[str, dict[str, Any]] = config["Displacements"]
        for displacement in self.displacements.values():
            displacement["data"] = external / displacement["data"]

        self.delays: dict[str, dict[str, Any]] = config["Delays"]
        for delay in self.delays.values():
            delay["data"] = external / delay["data"]

        # self.geo: dict[str, Any] = config["Geometric"]
        # self.geo["data"] = external / self.geo["data"]

        # self.tropo: dict[str, Any] = config["Tropospheric"]
        # self.tropo["data"] = external / self.tropo["data"]

        # self.iono: dict[str, Any] = config["Ionospheric"]
        # self.iono["data"] = external / self.iono["data"]

        # self.thermal: dict[str, Any] = config["ThermalDeformation"]
        # self.thermal["data"] = external / self.thermal["data"]

        # self.directories: dict[str, Path] = config["Directories"]
        # for key, val in self.directories.items():
        #     self.directories[key] = _base / val

        # # Models & switches
        # self.models = config["Models"]
        # self.switches: dict[str, bool] = config["Switches"]

        # # Delays
        # self.delays: dict[str, bool] = config["Delays"]

        # # Combine all tables
        # self.all: dict[str, Any] = (
        #     self.catalogues
        #     | self.ephemerides
        #     | self.metadata
        #     | self.directories
        #     | self.models
        #     | self.switches
        # )

        return None
