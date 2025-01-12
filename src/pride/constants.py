from astropy import time
from typing import Any, Literal
from .io import load_catalog, Setup
import numpy as np
from .logger import log

J2000: time.Time = time.Time("2000-01-01T12:00:00", scale="tt").utc  # type: ignore


# class Constants:
#     """Physical constants for a specific version of the JPL ephemerides

#     NOTE: Conversions between TDB, TCB and TT
#     mass_TCB = mass_TDB * (1 + L_B)
#     mass_TT = mass_TCB * (1 - L_C)
#     """

#     VALID_VERSIONS = Literal["421", "405", "403", "430", "440", "13c"]

#     def __init__(self, jpl_eph: VALID_VERSIONS) -> None:
#         """COPY PASTED FROM ORIGINAL CODE!"""

#         # jpl_eph used can be 405 or 421, the latter is the default
#         # math consts
#         self.CDEGRAD = 1.7453292519943296e-02
#         self.CARCRAD = 4.8481368110953599e-06
#         self.CTIMRAD = 7.2722052166430399e-05
#         self.SECDAY = 86400.0
#         self.JUL_CENT = 36525.0
#         # JULIAN DATE OF STANDARD EPOCH J2000.0
#         self.JD2000 = 2451545.0
#         # Algemeene Physical constants
#         self.C = 2.99792458e8
#         self.C_km = 2.99792458e5
#         self.F = 298.25642  # the tide-free value, IERS2010
#         self.AE = 6378136.3  # the tide-free value
#         self.J_2 = 1.0826e-3
#         self.AU = 149597870700.0  # DE430
#         self.TAUA = 499.0047838061
#         self.G = 6.67428e-11

#         # from tn36; also see http://ilrs.gsfc.nasa.gov/docs/2014/196C.pdf
#         self.L_B = 1.550519768e-8
#         self.L_C = 1.48082686741e-8
#         self.L_G = 6.969290134e-10

#         # DE/LE405 Header. TDB-compatible!!
#         if "403" in jpl_eph:
#             AU_DE405 = 1.49597870691000015e11  # m
#             self.GSUN = 0.295912208285591095e-03 * AU_DE405**3 / 86400.0**2
#             self.MU = 0.813005600000000044e02 ** (-1)
#             self.GEARTH = (
#                 0.899701134671249882e-09 * AU_DE405**3 / 86400.0**2 / (1 + self.MU)
#             )
#             self.GMOON = self.GEARTH * self.MU
#             self.GMPlanet = [
#                 0.491254745145081187e-10 * AU_DE405**3 / 86400.0**2,
#                 0.724345248616270270e-09 * AU_DE405**3 / 86400.0**2,
#                 0.954953510577925806e-10 * AU_DE405**3 / 86400.0**2,
#                 0.282534590952422643e-06 * AU_DE405**3 / 86400.0**2,
#                 0.845971518568065874e-07 * AU_DE405**3 / 86400.0**2,
#                 0.129202491678196939e-07 * AU_DE405**3 / 86400.0**2,
#                 0.152435890078427628e-07 * AU_DE405**3 / 86400.0**2,
#                 0.218869976542596968e-11 * AU_DE405**3 / 86400.0**2,
#             ]
#         elif "405" in jpl_eph:
#             AU_DE405 = 1.49597870691000015e11  # m
#             self.GSUN = 0.295912208285591095e-03 * AU_DE405**3 / 86400.0**2
#             self.MU = 0.813005600000000044e02 ** (-1)
#             self.GEARTH = (
#                 0.899701134671249882e-09 * AU_DE405**3 / 86400.0**2 / (1 + self.MU)
#             )
#             self.GMOON = self.GEARTH * self.MU
#             self.GMPlanet = [
#                 0.491254745145081187e-10 * AU_DE405**3 / 86400.0**2,
#                 0.724345248616270270e-09 * AU_DE405**3 / 86400.0**2,
#                 0.954953510577925806e-10 * AU_DE405**3 / 86400.0**2,
#                 0.282534590952422643e-06 * AU_DE405**3 / 86400.0**2,
#                 0.845971518568065874e-07 * AU_DE405**3 / 86400.0**2,
#                 0.129202491678196939e-07 * AU_DE405**3 / 86400.0**2,
#                 0.152435890078427628e-07 * AU_DE405**3 / 86400.0**2,
#                 0.218869976542596968e-11 * AU_DE405**3 / 86400.0**2,
#             ]
#         # DE/LE421 Header. TDB-compatible!!
#         elif "421" in jpl_eph:
#             AU_DE421 = 1.49597870699626200e11  # m
#             self.GSUN = 0.295912208285591100e-03 * AU_DE421**3 / 86400.0**2
#             self.MU = 0.813005690699153000e02 ** (-1)
#             self.GEARTH = (
#                 0.899701140826804900e-09 * AU_DE421**3 / 86400.0**2 / (1 + self.MU)
#             )
#             self.GMOON = self.GEARTH * self.MU
#             self.GMPlanet = [
#                 0.491254957186794000e-10 * AU_DE421**3 / 86400.0**2,
#                 0.724345233269844100e-09 * AU_DE421**3 / 86400.0**2,
#                 0.954954869562239000e-10 * AU_DE421**3 / 86400.0**2,
#                 0.282534584085505000e-06 * AU_DE421**3 / 86400.0**2,
#                 0.845970607330847800e-07 * AU_DE421**3 / 86400.0**2,
#                 0.129202482579265000e-07 * AU_DE421**3 / 86400.0**2,
#                 0.152435910924974000e-07 * AU_DE421**3 / 86400.0**2,
#                 0.217844105199052000e-11 * AU_DE421**3 / 86400.0**2,
#             ]
#         # DE/LE430 Header. TDB-compatible!!
#         elif "430" in jpl_eph:
#             AU_DE430 = 1.49597870700000000e11  # m
#             self.GSUN = 0.295912208285591100e-03 * AU_DE430**3 / 86400.0**2
#             self.MU = 0.813005690741906200e02 ** (-1)
#             self.GEARTH = (
#                 0.899701139019987100e-09 * AU_DE430**3 / 86400.0**2 / (1 + self.MU)
#             )
#             self.GMOON = self.GEARTH * self.MU
#             self.GMPlanet = [
#                 0.491248045036476000e-10 * AU_DE430**3 / 86400.0**2,
#                 0.724345233264412000e-09 * AU_DE430**3 / 86400.0**2,
#                 0.954954869555077000e-10 * AU_DE430**3 / 86400.0**2,
#                 0.282534584083387000e-06 * AU_DE430**3 / 86400.0**2,
#                 0.845970607324503000e-07 * AU_DE430**3 / 86400.0**2,
#                 0.129202482578296000e-07 * AU_DE430**3 / 86400.0**2,
#                 0.152435734788511000e-07 * AU_DE430**3 / 86400.0**2,
#                 0.217844105197418000e-11 * AU_DE430**3 / 86400.0**2,
#             ]
#         elif "440" in jpl_eph:
#             """https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de440_tech-comments.txt"""
#             AU_DE440 = 1.49597870700000000e11  # m
#             self.GSUN = 0.295912208284119560e-03 * AU_DE440**3 / 86400**2
#             self.MU = 0.813005682214972e02 ** (-1)
#             self.GEARTH = (
#                 0.899701139294734660e-09 * AU_DE440**3 / 86400**2 / (1 + self.MU)
#             )
#             self.GMOON = self.GEARTH * self.MU
#             self.GMPlanet = [
#                 0.491250019488931820e-10 * AU_DE440**3 / 86400**2,
#                 0.724345233264411870e-09 * AU_DE440**3 / 86400**2,
#                 0.954954882972581190e-10 * AU_DE440**3 / 86400**2,
#                 0.282534582522579170e-06 * AU_DE440**3 / 86400**2,
#                 0.845970599337629030e-07 * AU_DE440**3 / 86400**2,
#                 0.129202656496823990e-07 * AU_DE440**3 / 86400**2,
#                 0.152435734788519390e-07 * AU_DE440**3 / 86400**2,
#                 0.217509646489335810e-11 * AU_DE440**3 / 86400**2,
#             ]
#         # INPOP13c Header. TDB-compatible!!
#         elif "13c" in jpl_eph:
#             AU_13c = 1.495978707000000e11
#             self.GSUN = 0.2959122082912712e-03 * (AU_13c**3) / (86400.0**2)
#             self.MU = 0.8130056945994197e02 ** (-1)
#             self.GEARTH = (
#                 0.8997011572788968e-09 * AU_13c**3 / 86400.0**2 / (1 + self.MU)
#             )
#             self.GMOON = self.GEARTH * self.MU
#             self.GMPlanet = [
#                 0.4912497173300158e-10 * AU_13c**3 / 86400.0**2,
#                 0.7243452327305554e-09 * AU_13c**3 / 86400.0**2,
#                 0.9549548697395966e-10 * AU_13c**3 / 86400.0**2,
#                 0.2825345791109909e-06 * AU_13c**3 / 86400.0**2,
#                 0.8459705996177680e-07 * AU_13c**3 / 86400.0**2,
#                 0.1292024916910406e-07 * AU_13c**3 / 86400.0**2,
#                 0.1524357330444817e-07 * AU_13c**3 / 86400.0**2,
#                 0.2166807318808926e-11 * AU_13c**3 / 86400.0**2,
#             ]
#         else:
#             raise ValueError(f"Version {jpl_eph} of JPL ephemeris is not supported")

#         self.TDB_TCB = 1.0 + self.L_B  # F^-1
#         # G*masses in TCB-frame!
#         self.GM_TCB = (
#             np.hstack(
#                 [
#                     self.GMPlanet[0:2],
#                     self.GEARTH,
#                     self.GMPlanet[2:],
#                     self.GMOON,
#                     self.GSUN,
#                 ]
#             )
#             * self.TDB_TCB
#         )
#         # G*masses in TDB-frame!
#         self.GM = np.hstack(
#             [self.GMPlanet[0:2], self.GEARTH, self.GMPlanet[2:], self.GMOON, self.GSUN]
#         )

#         return None

#     @staticmethod
#     def from_setup(setup: "Setup") -> "Constants":

#         # Get path to meta-kernel file
#         __catalog = load_catalog("spacecraft.yaml")
#         __sc = str(setup.general["target"]).upper()
#         target: dict[str, Any] | None = None
#         for _target in __catalog.values():
#             if __sc in _target["names"]:
#                 target = _target
#                 break
#         if target is None:
#             raise ValueError("Specified target not in catalog: spacecraft.yaml")

#         kernel_path = setup.resources["ephemerides"] / target["short_name"]
#         metak_path = kernel_path / "metak.tm"

#         # Get version of ephemerides from meta-kernel
#         log.debug("LACKING ROBUST WAY TO GET VERSION OF EPHEMERIDES")
#         version: str = ""
#         with metak_path.open() as f:
#             for line in f:
#                 if "spk/DE" in line and ".BSP" in line:
#                     version = line.split("/")[-1].split(".")[0]
#                     break
#         if version == "":
#             raise ValueError("Failed to read ephemerides version from meta-kernel")

#         if version not in Constants.VALID_VERSIONS.__args__:
#             raise ValueError(f"Version {version} of JPL ephemeris is not supported")

#         return Constants(version)
