from beartype.typing import Tuple, Dict

from numpy import ndarray, newaxis
import numpy as np
from jaxtyping import Float
import xarray as xr

from .util import reshape_any_nextlast, reshape_halfoverlap_last, average_fast_to_slow


def process_level3(
    shear_segment: Float[ndarray, "n_shear time_fast"],
    pspd_segment: Float[ndarray, "time_fast"],
    fftlen: int,
    sampling_freq: float,
    spatial_response_wavenum: float,
    freq_highpass: float,
    chunklen: int,
    chunkoverlap: int,
    ancillary: Dict[str, Float[ndarray, "time_fast"]] = None,  # average to time_slow
):  # -> xr.Dataset: # beartype complains with BeartypeCallHintForwardRefException
    Pf, freq = spectra(shear_segment, fftlen, sampling_freq, chunklen, chunkoverlap)

    # Average to time_slow
    data_fast: Float[ndarray, "variable time_fast"] = (
        pspd_segment[:, newaxis]  # add dimension
        if ancillary is None
        else np.stack((pspd_segment, *(arr for k, arr in ancillary.items())), axis=0)
    )
    # platform speed
    data_slow: Float[ndarray, "variable time_slow"] = average_fast_to_slow(
        data_fast, fftlen, chunklen, chunkoverlap
    )
    pspda = data_slow[0, :]

    # to wavenumber domain
    Pk = Pf * pspda[newaxis, :, newaxis] / fftlen / (sampling_freq / 2)
    k: Float[ndarray, "time_slow k"] = freq[newaxis, :] / pspda[:, newaxis]

    # apply corrections
    Pk = apply_compensation_spatial_response(Pk, k, spatial_response_wavenum)
    Pk = apply_compensation_highpass(Pk, freq, freq_highpass)
    # apply_removal_coherent_vibrations(P)
    # get_uncertainty_estimates(P)

    data_vars = {
        "k": (["time_slow", "wavenumber"], k),
        "Pk": (["nshear", "time_slow", "wavenumber"], Pk),
        "Pf": (["nshear", "time_slow", "wavenumber"], Pf),
        "freq": (["wavenumber"], freq),
        "platform_speed": (["time_slow"], pspda),
    }
    data_vars.update(
        {
            name: (["time_slow"], data_slow[ind+1, :])
            for ind, name in enumerate(ancillary.keys())
        }
        if ancillary is not None
        else {}
    )

    return xr.Dataset(data_vars=data_vars)


def spectra(
    shear: Float[ndarray, "n_shear time_fast"],
    fftlen: int,
    sampling_freq: float,
    chunklen: int,
    chunkoverlap: int,
) -> Tuple[
    Float[ndarray, "n_shear segment freq"],
    Float[ndarray, "freq"],  # frequencies
]:
    """
    Produce spectra from cleaned shear time series"""
    # reshape
    # reshuffle time dimension into segments of length fftlen
    yr = reshape_halfoverlap_last(shear, fftlen)
    # subtract mean
    yr -= yr.mean(axis=-1)[..., newaxis]
    # hanning window
    yr *= np.hanning(fftlen)[newaxis, :]

    # periodograms
    freq = np.fft.rfftfreq(fftlen, d=1 / sampling_freq)
    Fyr = np.fft.rfft(yr)[:, :]
    Pf = (Fyr.conj() * Fyr).real
    # average spectra by chunks (reshape the segments)
    Pf = reshape_any_nextlast(Pf, chunklen, chunkoverlap).mean(axis=-2)

    return Pf, freq


def apply_compensation_spatial_response(
    x: Float[ndarray, "n_shear time_slow k"],
    k: Float[ndarray, "time_slow k"],
    k0: float,
) -> Float[ndarray, "n_shear time_slow k"]:
    correction_factor = 1.0 + (k / k0) ** 2
    return x * correction_factor[newaxis, :, :]


def apply_compensation_highpass(
    x: Float[ndarray, "n_shear time_slow f"],
    freq: Float[ndarray, "f"],
    freq_highpass: float,
) -> Float[ndarray, "n_shear time_slow f"]:
    correction_factor = 1.0 + (freq_highpass / freq**2) ** 2.0
    return x * correction_factor[newaxis, ...]
