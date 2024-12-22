import numpy as np

def interp(
    rjd: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    ut1: np.ndarray,
    n: int | None = None,
    rjd_int: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate EOPs"""
    ...
