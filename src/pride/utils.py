from functools import reduce
from .external.iers import interp
from astropy.utils import iers
import numpy as np
import erfa
from .logger import log
from astropy import time


def factors(n: int) -> list[int]:
    """Factorize a number

    :return: List of factors
    """
    facs = set(
        reduce(
            list.__add__,
            [[i, n // i] for i in range(1, int(n**0.5) + 1) if n % i == 0],
        )
    )
    return sorted(list(facs))


class EOP:

    def __init__(self, bulletin: str = "B") -> None:

        # Select EOP bulletin
        match bulletin:
            case "A":
                eop_table = iers.IERS_A.open()
            case "B":
                eop_table = iers.IERS_B.open()
            case _:
                log.error(
                    f"Failed to initialize EOPs: Invalid bulletin {bulletin}"
                )
                exit(1)

        # Extract EOP data
        self.eops = {
            mjd: np.array([xp, yp, ut1_utc, dx, dy])
            for mjd, ut1_utc, xp, yp, dx, dy in zip(
                eop_table["MJD"].data,
                eop_table["UT1_UTC"].data,
                eop_table["PM_x"].to("rad").data,
                eop_table["PM_y"].to("rad").data,
                eop_table["dX_2000A"].to("rad").data,
                eop_table["dY_2000A"].to("rad").data,
            )
        }

        return None

    def at_epoch(self, epoch: time.Time) -> np.ndarray:

        # Get MJD for given epochs
        mjd_int_list: np.ndarray = np.array([epoch.mjd // 1], dtype=int).ravel()  # type: ignore
        mjd_list = np.array(epoch.mjd)

        # Output container
        out = np.zeros((len(mjd_int_list), 5))

        for idx, (mjd_int, mjd) in enumerate(zip(mjd_int_list, mjd_list)):

            # Define a range of seven days around epoch
            mjd_range = np.array([mjd_int + i for i in range(-3, 4)])

            # Get default values of EOPs for range of epochs
            ut1_utc, xp, yp, dx, dy = np.array(
                [self.eops[mjd_int + i] for i in range(-3, 4)]
            ).T

            # Interpolate X, Y and UT1-UTC
            x, y, ut1_utc = interp.interp(
                mjd_range, xp, yp, ut1_utc, rjd_int=mjd
            )

            # Interpolate dX and dY
            dx = interp.lagint(mjd_range, dx, xint=mjd)
            dy = interp.lagint(mjd_range, dy, xint=mjd)

            # Store interpolated values
            out[idx] = np.array([ut1_utc, x, y, dx, dy])

        return out


def gcrs_2_itrf(eops: np.ndarray, epoch: time.Time) -> np.ndarray:

    # Epochs in TT and UTC
    tt_epoch: time.Time = epoch.tt  # type: ignore
    utc_epoch: time.Time = epoch.utc  # type: ignore

    # EOPS
    ut1_utc, xp, yp, dx, dy = eops

    # Compute matrix following algorithm from IERS Conventions 2010 (5.9)
    x, y = erfa.xy06(tt_epoch.jd1, tt_epoch.jd2)
    s = erfa.s06(tt_epoch.jd1, tt_epoch.jd2, x, y)
    RC2I_matrix = erfa.c2ixys(x + dx, y + dy, s)
    era = erfa.era00(utc_epoch.jd1, utc_epoch.jd2 + (ut1_utc / 86400.0))
    RC2TI_matrix = erfa.rz(era, RC2I_matrix)
    s_prime = erfa.sp00(tt_epoch.jd1, tt_epoch.jd2)
    RPOM_matrix = erfa.pom00(xp, yp, s_prime)

    return RPOM_matrix @ RC2TI_matrix


def itrf_2_gcrs(eops: np.ndarray, epoch: time.Time) -> np.ndarray:

    return gcrs_2_itrf(eops, epoch).swapaxes(1, 2)
