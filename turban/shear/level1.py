from dataclasses import dataclass
import numpy as np
from turban.util import butterfilt, fft_grad

try:
    import xarray as xr
except ImportError:
    xr = None
from netCDF4 import Dataset
from numpy import ndarray
import numpy as np
from jaxtyping import Float
from beartype.typing import Tuple

from turban.shear.config import ShearConfig

sea_water_density = 1025.0

@dataclass
class ShearLevel1:

    pspd: Float[ndarray, "time"]
    shear: Float[ndarray, "n_shear time"]
    cfg: ShearConfig

    @classmethod
    def from_raw(cls, pressure_raw, shear_raw, cfg: ShearConfig):
        pspd, pressure_lp = get_vsink(pressure_raw, cfg.sampling_freq)
        shear_phys = process_level1(pspd, shear_raw)
        return cls(
            pspd=pspd,
            shear_phys=shear_phys,
            cfg=cfg,
        )

    @classmethod
    def from_atomix_netcdf(cls, fname: str):
        ds = xr.load_dataset(fname, group="L1_converted")
        return cls(
            pspd=ds.PSPD_REL.values,
            shear=ds.SHEAR.values,
            cfg=ShearConfig.from_atomix_netcdf(fname),
        )


def process_level1(
    vsink: Float[ndarray, "time"],
    shear: Float[ndarray, "n_shear time"],
) -> Float[ndarray, "n_shear time"]:
    """
    Convert to physical units"""
    shear_phys = mss_shear_physical(vsink, shear)
    return shear_phys


def get_vsink(pressure_raw, sampling_freq=1024.0):
    # lowpass filter pressure
    pressure_lp = butterfilt(
        signal=pressure_raw,
        cutoff_freq_Hz=0.5,
        sampling_freq=sampling_freq,
        btype="low",
    )
    # sinking speed
    vsink = fft_grad(pressure_lp, 1 / sampling_freq)
    return vsink, pressure_lp


