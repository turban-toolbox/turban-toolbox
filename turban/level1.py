import numpy as np
from .util import butterfilt, fft_grad
from numpy import ndarray
import numpy as np
from jaxtyping import Float
from beartype.typing import Tuple

sea_water_density = 1025.0


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


def mss_shear_physical(
    vsink: Float[ndarray, "time"],
    shear_channels: Float[ndarray, "n_shear time"],
    sampling_freq=1024.0,
) -> Float[ndarray, "n_shear time"]:
    """Calculates physical shear for the MSS shear probes.
    Returns:
        Low-pass filtered pressure
        Sinking speed
        shear channels in physical units, in same order as passed in

    TODO: data needs canonical column names
    """
    target_freq = 256.0

    # convert from MSS "shear" (i.e. a velocity) to time derivative (units m/s/s)
    shear_channels_phys = []
    for i, sh in enumerate(shear_channels):
        sh_phys = fft_grad(sh, 1 / sampling_freq) / vsink**2 / sea_water_density
        shear_channels_phys.append(sh_phys)

    return np.array(shear_channels_phys)
