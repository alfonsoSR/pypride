from typing import TYPE_CHECKING
import datetime
from astropy import time
from ..logger import log
from ..types import SourceType
import spiceypy as spice
from nastro import types as nt
from ..constants import J2000
import numpy as np

if TYPE_CHECKING:
    from .experiment import Experiment
    from .source import Source
    from ..types import Band
    from .baseline import Baseline


class Observation:
    """Observation of a source

    The time stamps are stored as an astropy Time object including information about the coordinates of the station for which the observation was performed. The coordinates of the station are obtained by correcting the location at a reference epoch (loaded from data file) for tectonic motion.

    :param source: Source object representing the target
    :param band: Frequency band in which the source was detected
    :param tstamps: Collection of epochs in which the source was detected
    """

    def __init__(
        self,
        baseline: "Baseline",
        source: "Source",
        band: "Band",
        tstamps: list[datetime.datetime],
    ) -> None:
        """Initialize observation

        :param Baseline: Baseline from which the observation was performed
        :param source: Source object representing the target
        :param band: Frequency band in which the source was detected
        :param tstamps: List of UTC epochs in which the source was detected
        :param exp: Experiment object to which the observation belongs
        """

        self.source = source
        self.band = band
        _tstamps = time.Time(sorted(tstamps), scale="utc")
        self.tstamps = time.Time(
            sorted(tstamps),
            scale="utc",
            location=baseline.station.tectonic_corrected_location(_tstamps),
        )
        self.baseline = baseline

        # Optional attributes
        setattr(self, "_exp", getattr(baseline.station, "_exp"))
        setattr(self, "__icrf2itrf", None)
        setattr(self, "__seu2itrf", None)

        return None

    @property
    def exp(self) -> "Experiment":

        if getattr(self, "_exp") is None:
            log.error(f"Experiment not set for {self.source.name} observation")
            exit(1)
        return getattr(self, "_exp")

    @property
    def icrf2itrf(self) -> "np.ndarray":

        if getattr(self, "__icrf2itrf") is None:
            log.error(
                f"ICRF to ITRF transformation not found for {self.source.name}"
            )
            exit(0)
        return getattr(self, "__icrf2itrf")

    @property
    def seu2itrf(self) -> "np.ndarray":

        if getattr(self, "__seu2itrf") is None:
            log.error(
                f"SEU to ITRF transformation not found for {self.source.name}"
            )
            exit(0)
        return getattr(self, "__seu2itrf")

    def source_azimuth_and_elevation(self):

        match self.source.type:
            case SourceType.NearField:
                return self.near_field_azel()
            case SourceType.FarField:
                print(f"{self.source.name} is a calibrator")
                return None
                # raise NotImplementedError("Far-field sources not supported")
            case _:
                log.error(
                    "Failed to calculate spherical coordinates for "
                    f"{self.source.name}: Invalid source type"
                )
                exit(1)

        return None

    def near_field_azel(self) -> tuple[np.ndarray, np.ndarray]:

        clight = spice.clight() * 1e3

        # Calculate station coordinates at observation epochs
        xsta_gcrf = self.baseline.station.location(self.tstamps, frame="icrf")
        cstate_sta = nt.CartesianState(*xsta_gcrf.T, *(0.0 * xsta_gcrf.T))

        # Initialize light-time between target and station
        et: np.ndarray = (
            (self.tstamps.tdb - J2000.tdb).to("s").value  # type: ignore
        )
        xsc_gcrf = (
            np.array(
                spice.spkpos(
                    self.source.spice_id, et, "J2000", "NONE", "EARTH"
                )[0]
            )
            * 1e3
        )
        lt_0 = np.linalg.norm(xsc_gcrf - xsta_gcrf, axis=1) / clight
        epoch_tx0 = self.tstamps.tdb - time.TimeDelta(lt_0, format="sec")

        # Initialize light-time container
        lt_i = np.zeros_like(lt_0)
        ni = 0
        precision = 1e-12
        nmax = 3

        # Get GM of celestial bodies
        bodies = self.exp.setup.internal["lt_correction_bodies"]
        bodies_gm = (
            np.array([spice.bodvrd(body, "GM", 1)[1][0] for body in bodies])
            * 1e9
        )

        while np.any(np.abs(lt_0 - lt_i) > precision) and (ni < nmax):

            # Update light-time
            lt_i = lt_0
            epoch_tx = self.tstamps.tdb - time.TimeDelta(lt_i, format="sec")
            dt = (epoch_tx.tdb - epoch_tx0.tdb).to("s").value  # type: ignore

            # Calculate target's position wrt station at TX epoch
            et = (epoch_tx.tdb - J2000.tdb).to("s").value  # type: ignore
            s_sc_earth_tx = nt.CartesianState(
                *np.array(
                    spice.spkezr(
                        self.source.spice_id, et, "J2000", "NONE", "EARTH"
                    )[0]
                ).T
                * 1e3
            )
            s_sta_sc_tx = cstate_sta - s_sc_earth_tx
            r01 = s_sta_sc_tx.r_vec.T
            r01_mag = s_sta_sc_tx.r_mag
            r0 = s_sc_earth_tx.r_vec.T
            v0 = s_sc_earth_tx.v_vec.T

            # Get state of celestial bodies at TX epoch
            bodies_cstate = np.array(
                [
                    np.array(spice.spkezr(body, et, "J2000", "NONE", "SSB")[0])
                    * 1e3
                    for body in bodies
                ]
            )
            rb = bodies_cstate[:, :, :3]
            vb = bodies_cstate[:, :, 3:]

            # Calculate terms of relativistic correction
            r0b = r0[None, :, :] - (rb - dt[None, :, None] * vb)
            r0b_mag = np.linalg.norm(r0b, axis=-1)
            r1b = xsta_gcrf[None, :, :] - rb
            r1b_mag = np.linalg.norm(r1b, axis=-1)
            r01b_mag = np.linalg.norm(r1b - r0b, axis=-1)

            # Calculate relativistic correction
            gmc = 2.0 * bodies_gm[:, None] / (clight * clight)
            rlt = np.sum(
                (gmc / clight)
                * np.log(
                    (r0b_mag + r1b_mag + r01b_mag + gmc)
                    / (r0b_mag + r1b_mag - r01b_mag + gmc)
                ),
                axis=0,
            )

            # Update light-time with relativistic correction
            denom = 1.0 - (np.sum((r01.T * v0.T), axis=0) / (clight * r01_mag))
            lt_0 -= (lt_0 - r01_mag / clight - rlt) / denom

            # Update TX epoch and counter
            epoch_tx = self.tstamps.tdb - time.TimeDelta(lt_0, format="sec")
            ni += 1

        # Calculate position of target wrt station at TX epoch [Final]
        et: np.ndarray = (epoch_tx.tdb - J2000.tdb).to("s").value  # type: ignore
        s_sc_earth_tx = nt.CartesianState(
            *np.array(
                spice.spkezr(
                    self.source.spice_id, et, "J2000", "NONE", "EARTH"
                )[0]
            ).T
            * 1e3
        )
        s_sc_sta_tx = s_sc_earth_tx - cstate_sta

        # Calculate azimuth and elevation
        uvec_icrf = (s_sc_earth_tx - cstate_sta).r_uvec.T
        uvec_itrf = self.icrf2itrf.swapaxes(-1, -2) @ uvec_icrf[:, :, None]
        uvec_seu = (self.seu2itrf.swapaxes(-1, -2) @ uvec_itrf).squeeze().T

        el = np.arcsin(uvec_seu[0])
        az = np.arctan2(uvec_seu[1], uvec_seu[2])
        az += (az < 0.0) * 2.0 * np.pi

        return az, el
