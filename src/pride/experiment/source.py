from typing import TYPE_CHECKING
from ..types import SourceType
from astropy import coordinates
from ..logger import log

if TYPE_CHECKING:
    from .experiment import Experiment
    from ..io import VexContent


class Source:

    def __init__(
        self,
        name: str,
        source_type: SourceType,
        ra: float,
        dec: float,
        spice_id: str | None,
    ) -> None:

        self.name = name
        self.type = source_type
        self.coordinates = coordinates.SkyCoord(
            ra, dec, unit="rad", frame="icrs"
        )
        self.spice_id = spice_id
        setattr(self, "_exp", None)

        return None

    @property
    def exp(self) -> "Experiment":

        if getattr(self, "_exp") is None:
            log.error(f"Experiment not set for {self.name} source")
            exit(1)
        return getattr(self, "_exp")

    @staticmethod
    def from_experiment(name: str, experiment: "Experiment") -> "Source":

        # Read source information from VEX file
        if name not in experiment.vex["SOURCE"]:
            log.error(
                f"Failed to generate source: {name} not found in VEX file"
            )
            exit(1)
        source_info = experiment.vex["SOURCE"][name]

        # Read coordinates from VEX
        if source_info["ref_coord_frame"] != "J2000":
            raise NotImplementedError(
                f"Failed to generate source object for {name}: "
                f"Coordinate frame {source_info['ref_coord_frame']} not supported"
            )
        coords = coordinates.SkyCoord(
            source_info["ra"], source_info["dec"], frame="icrs"
        )

        # Identify source type and add spice ID if necessary
        source_type = Source.__find_type(source_info, name)
        spice_id: str | None = None
        if source_type == SourceType.FarField:
            spice_id = experiment.setup.general["target"]

        # Generate object and add experiment as attribute
        source = Source(
            name,
            source_type,
            coords.ra.rad,  # type: ignore
            coords.dec.rad,  # type: ignore
            spice_id,
        )
        setattr(source, "_exp", experiment)

        return source

    @staticmethod
    def __find_type(source_info: "VexContent", _source_name: str) -> SourceType:

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

    def __repr__(self) -> str:
        return f"{self.name:10s}: {super().__repr__()}"
