from .core import Displacement
from typing import Any, TYPE_CHECKING
from ..io import internal_file
from ..logger import log
import numpy as np
from astropy import time
import spiceypy as spice
from ..constants import J2000, Constants
from nastro import types as nt
import erfa

if TYPE_CHECKING:
    from ..experiment.core import Station


class OceanLoading(Displacement):
    """Displacement due to ocean loading"""

    name = "OceanLoading"
    etc = {}
    models = ["tpxo72"]

    def ensure_resources(self) -> None:

        source = self.config["data"] / f"{self.config['model']}.blq"
        if not source.exists():
            log.error(
                f"Failed to initialize {self.name} displacement: {source} not found"
            )
            log.info("Downloading ocean loading data will be supported in the future")
            exit(1)

        self._resources["source"] = source

        return None

    def load_resources(self) -> None:

        with self._resources["source"].open("r") as f:

            content = f.readlines()

            for baseline in self.exp.baselines:

                _amp: np.ndarray | None = None
                _phs: np.ndarray | None = None
                for idx, line in enumerate(content):
                    line = line.strip()
                    if len(line) == 0 or line[0] == "$":
                        continue
                    if any([name in line for name in baseline.station.possible_names]):
                        idx += 1
                        while content[idx][0] == "$":
                            idx += 1
                        line = content[idx].strip()
                        _amp = (
                            np.array(
                                " ".join(content[idx : idx + 3]).split(), dtype=float
                            )
                            .reshape((3, 11))
                            .T
                        )
                        _phs = (
                            np.array(
                                " ".join(content[idx + 3 : idx + 6]).split(),
                                dtype=float,
                            )
                            .reshape((3, 11))
                            .T
                        )
                        break

                if _amp is None or _phs is None:
                    log.error(
                        f"Failed to load ocean loading data for {baseline.station}"
                    )
                    exit(1)
                self.resources[baseline.station.name] = (_amp, _phs)

        return None


class SolidTide(Displacement):
    """Displacement due to solid Earth tides induced by the Sun and the Moon"""

    name = "SolidTide"
    etc = {}
    models = ["dehan_inel"]
    requires_spice = True

    def ensure_resources(self) -> None:

        self.resources["constants"] = {
            "Re": 6378136.6,
            "MRsun": 332946.0482,
            "MRmoon": 0.0123000371,
        }
        self.resources["inphase"] = {
            "h0": 0.6078,
            "h2": -0.0006,
            "l0": 0.0847,
            "l2": 0.0002,
            "h3": 0.292,
            "l3": 0.015,
        }
        self.resources["outphase"] = {}
        self.resources["outphase"]["diurnal"] = {
            "hi": -0.0025,
            "li": -0.0007,
        }
        self.resources["outphase"]["semidiurnal"] = {
            "hi": -0.0022,
            "li": -0.0007,
        }

        self.resources["transverse"] = {
            "diurnal": {"l1": 0.0012},
            "semidiurnal": {"l1": 0.0024},
        }

        # Step 2 - Diurnal
        self.resources["step2d"] = {}

        return None

    def load_resources(self) -> None:

        log.debug(f"{self.name} displacement does not require external resources")

        return None

    def calculate(self, station: "Station", epochs: "time.Time") -> None:

        # Get epochs as ET
        et = (epochs.tdb - J2000.tdb).sec  # type: ignore

        # State vectors of station, Sun and Moon in ITRF
        _pos = np.array(station.location(epochs).geocentric)
        sta = nt.CartesianState(*_pos, *(0.0 * _pos))
        sun = nt.CartesianState(
            *np.array(spice.spkezr("sun", et, "ITRF93", "CN", "earth")[0]).T
        )
        moon = nt.CartesianState(
            *np.array(spice.spkezr("moon", et, "ITRF93", "CN", "earth")[0]).T
        )

        # Corrections in the time domain: IERS 2010 - Step 1
        # Displacement due to second and third degree tides
        disp = self.in_phase(sta, sun, moon)

        # Out of phase contributions from imaginary parts of h2 and l2
        disp += self.out_of_phase(sta, sun, moon)

        # Correction due to dependence of Love numbers on latitude
        disp += self.transverse_l1(sta, sun, moon)

        # Corrections in the frequency domain: IERS 2010 - Step 2

        return None

    def in_phase(
        self,
        sta: nt.CartesianState,
        sun: nt.CartesianState,
        moon: nt.CartesianState,
    ) -> np.ndarray:
        """"""

        # Parameters for in-phase correction [p103 - IERS 2010]
        params = self.resources["inphase"]
        const = self.resources["constants"]

        # Adjust Love numbers for station position
        sinphi2 = 1.0 - (sta.r_uvec[0] * sta.r_uvec[0] + sta.r_uvec[1] * sta.r_uvec[1])
        h2 = params["h0"] + 0.5 * params["h2"] * (3.0 * sinphi2 - 1.0)
        l2 = params["l0"] + 0.5 * params["l2"] * (3.0 * sinphi2 - 1.0)
        h3 = params["h3"]
        l3 = params["l3"]

        # Order 2 displacement
        ratio_sun2 = const["MRsun"] * const["Re"] * ((const["Re"] / sun.r_mag) ** 3)
        cos_sun_sta = np.sum(sta.r_uvec * sun.r_uvec, axis=0)
        term_h2_sun = h2 * sta.r_uvec * (1.5 * cos_sun_sta * cos_sun_sta - 0.5)
        term_l2_sun = 3.0 * l2 * cos_sun_sta * (sun.r_uvec - sta.r_uvec * cos_sun_sta)
        disp2_sun = ratio_sun2 * (term_h2_sun + term_l2_sun)

        ratio_moon2 = const["MRmoon"] * const["Re"] * ((const["Re"] / moon.r_mag) ** 3)
        cos_moon_sta = np.sum(sta.r_uvec * moon.r_uvec, axis=0)
        term_h2_moon = h2 * sta.r_uvec * (1.5 * cos_moon_sta * cos_moon_sta - 0.5)
        term_l2_moon = (
            3.0 * l2 * cos_moon_sta * (moon.r_uvec - sta.r_uvec * cos_moon_sta)
        )
        disp2_moon = ratio_moon2 * (term_h2_moon + term_l2_moon)

        # Order 3 displacement
        ratio_sun3 = ratio_sun2 * const["Re"] / sun.r_mag
        term_h3_sun = h3 * sta.r_uvec * (2.5 * (cos_sun_sta**3) - 1.5 * cos_sun_sta)
        term_l3_sun = (
            l3
            * (7.5 * (cos_sun_sta**2) - 1.5)
            * (sun.r_uvec - sta.r_uvec * cos_sun_sta)
        )
        disp3_sun = ratio_sun3 * (term_h3_sun + term_l3_sun)

        ratio_moon3 = ratio_moon2 * const["Re"] / moon.r_mag
        term_h3_moon = h3 * sta.r_uvec * (2.5 * (cos_moon_sta**3) - 1.5 * cos_moon_sta)
        term_l3_moon = (
            l3
            * (7.5 * (cos_moon_sta**2) - 1.5)
            * (moon.r_uvec - sta.r_uvec * cos_moon_sta)
        )
        disp3_moon = ratio_moon3 * (term_h3_moon + term_l3_moon)

        return disp2_sun + disp2_moon + disp3_sun + disp3_moon

    def out_of_phase(
        self,
        sta: nt.CartesianState,
        sun: nt.CartesianState,
        moon: nt.CartesianState,
    ) -> np.ndarray:
        """"""

        # Parameters for corrections [p103 - IERS 2010]
        dlove = self.resources["outphase"]["diurnal"]
        slove = self.resources["outphase"]["semidiurnal"]
        const = self.resources["constants"]

        # Station latitude and longitude
        sinlat = sta.r_uvec[2]
        coslat = np.sqrt(1.0 - sinlat * sinlat)
        sinlon = sta.r_uvec[1] / coslat
        coslon = sta.r_uvec[0] / coslat
        cos2lat = coslat * coslat - sinlat * sinlat
        cos2lon = coslon * coslon - sinlon * sinlon
        sin2lon = 2.0 * sinlon * coslon

        # Mass ratios
        ratio2_sun = const["MRsun"] * const["Re"] * ((const["Re"] / sun.r_mag) ** 3)
        ratio2_moon = const["MRmoon"] * const["Re"] * ((const["Re"] / moon.r_mag) ** 3)

        # Radial displacement
        # Diurnal
        dr_sun = (
            -3.0
            * dlove["hi"]
            * sinlat
            * coslat
            * ratio2_sun
            * sun.r_uvec[2]
            * (sun.r_uvec[0] * sinlon - sun.r_uvec[1] * coslon)
        )
        dr_moon = (
            -3.0
            * dlove["hi"]
            * sinlat
            * coslat
            * ratio2_moon
            * moon.r_uvec[2]
            * (moon.r_uvec[0] * sinlon - moon.r_uvec[1] * coslon)
        )
        dr = dr_sun + dr_moon

        # Semidiurnal
        sr_sun = (
            -0.75
            * slove["hi"]
            * ratio2_sun
            * (coslat * coslat)
            * (
                sin2lon * (sun.r_uvec[0] ** 2 + sun.r_uvec[1] ** 2)
                - 2.0 * cos2lon * sun.r_uvec[0] * sun.r_uvec[1]
            )
        )
        sr_moon = (
            -0.75
            * slove["hi"]
            * ratio2_moon
            * (coslat * coslat)
            * (
                sin2lon * (moon.r_uvec[0] ** 2 + moon.r_uvec[1] ** 2)
                - 2.0 * cos2lon * moon.r_uvec[0] * moon.r_uvec[1]
            )
        )
        sr = sr_sun + sr_moon

        # Eastward displacement
        # Diurnal
        de_sun = (
            -3.0
            * dlove["li"]
            * sinlat
            * ratio2_sun
            * sun.r_uvec[2]
            * (sun.r_uvec[0] * coslon + sun.r_uvec[1] * sinlon)
        )
        de_moon = (
            -3.0
            * dlove["li"]
            * sinlat
            * ratio2_moon
            * moon.r_uvec[2]
            * (moon.r_uvec[0] * coslon + moon.r_uvec[1] * sinlon)
        )
        de = de_sun + de_moon

        # Semidiurnal
        se_sun = (
            -1.5
            * slove["li"]
            * ratio2_sun
            * coslat
            * (
                cos2lon * (sun.r_uvec[0] ** 2 - sun.r_uvec[1] ** 2)
                + 2.0 * sin2lon * sun.r_uvec[0] * sun.r_uvec[1]
            )
        )
        se_moon = (
            -1.5
            * slove["li"]
            * ratio2_moon
            * coslat
            * (
                cos2lon * (moon.r_uvec[0] ** 2 - moon.r_uvec[1] ** 2)
                + 2.0 * sin2lon * moon.r_uvec[0] * moon.r_uvec[1]
            )
        )
        se = se_sun + se_moon

        # Northward displacement
        # Diurnal
        dn_sun = (
            -3.0
            * dlove["li"]
            * cos2lat
            * ratio2_sun
            * sun.r_uvec[2]
            * (sun.r_uvec[0] * sinlon - sun.r_uvec[1] * coslon)
        )
        dn_moon = (
            -3.0
            * dlove["li"]
            * cos2lat
            * ratio2_moon
            * moon.r_uvec[2]
            * (moon.r_uvec[0] * sinlon - moon.r_uvec[1] * coslon)
        )
        dn = dn_sun + dn_moon

        # Semidiurnal
        sn_sun = (
            1.5
            * slove["li"]
            * ratio2_sun
            * sinlat
            * coslat
            * (
                sin2lon * (sun.r_uvec[0] ** 2 - sun.r_uvec[1] ** 2)
                - 2.0 * cos2lon * sun.r_uvec[0] * sun.r_uvec[1]
            )
        )
        sn_moon = (
            1.5
            * slove["li"]
            * ratio2_moon
            * sinlat
            * coslat
            * (
                sin2lon * (moon.r_uvec[0] ** 2 - moon.r_uvec[1] ** 2)
                - 2.0 * cos2lon * moon.r_uvec[0] * moon.r_uvec[1]
            )
        )
        sn = sn_sun + sn_moon

        # Correction vector
        diurnal = np.array(
            [
                dr * coslon * coslat - de * sinlon - dn * sinlat * coslon,
                dr * sinlon * coslat + de * coslon - dn * sinlat * sinlon,
                dr * sinlat + dn * coslat,
            ]
        )
        semidiurnal = np.array(
            [
                sr * coslon * coslat - se * sinlon - sn * sinlat * coslon,
                sr * sinlon * coslat + se * coslon - sn * sinlat * sinlon,
                sr * sinlat + sn * coslat,
            ]
        )

        return diurnal + semidiurnal

    def transverse_l1(
        self,
        sta: nt.CartesianState,
        sun: nt.CartesianState,
        moon: nt.CartesianState,
    ) -> np.ndarray:

        # Parameters for corrections [p103 - IERS 2010]
        const = self.resources["constants"]
        dl1 = self.resources["transverse"]["diurnal"]["l1"]
        sl1 = self.resources["transverse"]["semidiurnal"]["l1"]

        # Station latitude and longitude
        sinlat = sta.r_uvec[2]
        coslat = np.sqrt(1.0 - sinlat * sinlat)
        sinlon = sta.r_uvec[1] / coslat
        coslon = sta.r_uvec[0] / coslat
        cos2lon = coslon * coslon - sinlon * sinlon
        sin2lon = 2.0 * sinlon * coslon
        cos2lat = coslat * coslat - sinlat * sinlat

        # Mass ratios
        ratio_sun = const["MRsun"] * const["Re"] * ((const["Re"] / sun.r_mag) ** 3)
        ratio_moon = const["MRmoon"] * const["Re"] * ((const["Re"] / moon.r_mag) ** 3)

        # Northward displacement
        # Diurnal
        dn_sun = (
            -3.0
            * dl1
            * ratio_sun
            * (sinlat**2)
            * sun.r_uvec[2]
            * (sun.r_uvec[0] * coslon + sun.r_uvec[1] * sinlon)
        )
        dn_moon = (
            -3.0
            * dl1
            * ratio_moon
            * (sinlat**2)
            * moon.r_uvec[2]
            * (moon.r_uvec[0] * coslon + moon.r_uvec[1] * sinlon)
        )
        dn = dn_sun + dn_moon

        # Semidiurnal
        sn_sun = (
            -1.5
            * sl1
            * ratio_sun
            * sinlat
            * coslat
            * (
                cos2lon * (sun.r_uvec[0] ** 2 - sun.r_uvec[1] ** 2)
                + 2.0 * sin2lon * sun.r_uvec[0] * sun.r_uvec[1]
            )
        )
        sn_moon = (
            -1.5
            * sl1
            * ratio_moon
            * sinlat
            * coslat
            * (
                cos2lon * (moon.r_uvec[0] ** 2 - moon.r_uvec[1] ** 2)
                + 2.0 * sin2lon * moon.r_uvec[0] * moon.r_uvec[1]
            )
        )
        sn = sn_sun + sn_moon

        # Eastward displacement
        # Diurnal
        de_sun = (
            -3.0
            * dl1
            * ratio_sun
            * sinlat
            * cos2lat
            * sun.r_uvec[2]
            * (sun.r_uvec[0] * sinlon - sun.r_uvec[1] * coslon)
        )
        de_moon = (
            -3.0
            * dl1
            * ratio_moon
            * sinlat
            * cos2lat
            * moon.r_uvec[2]
            * (moon.r_uvec[0] * sinlon - moon.r_uvec[1] * coslon)
        )
        de = de_sun + de_moon

        # Semidiurnal
        se_sun = (
            -1.5
            * sl1
            * ratio_sun
            * (sinlat**2)
            * coslat
            * (
                sin2lon * (sun.r_uvec[0] ** 2 - sun.r_uvec[1] ** 2)
                - 2.0 * cos2lon * sun.r_uvec[0] * sun.r_uvec[1]
            )
        )
        se_moon = (
            -1.5
            * sl1
            * ratio_moon
            * (sinlat**2)
            * coslat
            * (
                sin2lon * (moon.r_uvec[0] ** 2 - moon.r_uvec[1] ** 2)
                - 2.0 * cos2lon * moon.r_uvec[0] * moon.r_uvec[1]
            )
        )
        se = se_sun + se_moon

        # Correction vector
        diurnal = np.array(
            [
                -de * sinlon - dn * sinlat * coslon,
                de * coslon - dn * sinlat * sinlon,
                dn * coslat,
            ]
        )
        semidiurnal = np.array(
            [
                -se * sinlon - sn * sinlat * coslon,
                se * coslon - sn * sinlat * sinlon,
                sn * coslat,
            ]
        )

        return diurnal + semidiurnal
