from beartype.typing import Tuple

import gsw
import numpy as np
from numpy import ndarray
from jaxtyping import Float, Int

from .temperature import *


def calc_ctd(
    C: Float[ndarray, "time"],
    T: Float[ndarray, "time"],
    P: Float[ndarray, "time"],
    lon: float,
    lat: float,
) -> Tuple[
    Float[ndarray, "time"],
    Float[ndarray, "time"],
    Float[ndarray, "time"],
]:
    """
    Parameters
    ----------
        C : conductivity (mS/cm)
        T : in situ temperature (deg C, ITS-90)
        P : Pressure (dbar)
        lon, lat: degrees East/North

    Returns
    -------
        sigma0 : potential density anomaly (kg/m3)
        SA : Absolute salinity, g/kg
        CT : conservative temperature (deg C)
    """
    SP = gsw.SP_from_C(C, T, P)
    SA = gsw.SA_from_SP(SP, P, lon, lat)
    CT = gsw.CT_from_t(SA, T, P)
    sigma0 = gsw.sigma0(SA, CT)
    return (sigma0, SA, CT)


def fofonoff_filt(x: Float[ndarray, "time"], tau: int) -> Float[ndarray, "time"]:
    """Correct time series for lagged sensor response. For MSS temperature, an
    OK default value is 55 but this should be determined individually"""
    w = 200
    w2 = w // 2
    # pad with nans
    y: Float[ndarray, "agg window"] = reshape_halfoverlap_last(x, w)
    # dummy time vector in seconds
    samples = np.arange(w)
    t: Int[ndarray, "agg2"] = np.arange(0, len(x) + 1, w2)

    dxdt_agg: Float[ndarray, "agg2"] = np.r_[
        np.polyfit(x=samples[:w2], y=y[0, :w2], deg=1)[0],  # first bit
        np.polyfit(x=samples, y=y.transpose(), deg=1)[0, :],
        np.polyfit(x=samples[:w2], y=y[-1, -w2:], deg=1)[0],  # last bit
    ]
    dxdt: Float[ndarray, "time"] = np.interp(np.arange(len(x)), t, dxdt_agg)
    return (dxdt * tau) + x
