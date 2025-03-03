from dataclasses import dataclass

from beartype.typing import Tuple, Dict

from numpy import ndarray, newaxis
import numpy as np
from jaxtyping import Float, Int
import xarray as xr
from netCDF4 import Dataset

from turban.util import reshape_any_nextlast, reshape_halfoverlap_last, average_fast_to_slow
from turban.level1 import ShearLevel1
from turban.level2 import ShearLevel2, split_data
from turban.config import ShearConfig


@dataclass
class ShearLevel3:
    Pk: Float[ndarray, "nshear time wavenumber"]
    k: Float[ndarray, "time wavenumber"]
    Pf: Float[ndarray, "nshear time wavenumber"]
    freq: Float[ndarray, "wavenumber"]
    platform_speed: Float[ndarray, "time"]
    cfg: ShearConfig

    @classmethod
    def from_level2(
        cls,
        level2: ShearLevel2,
    ) -> "ShearLevel3":
        k, Pk, Pf, freq, platform_speed, ancillary = process_level3(
            shear=level2.shear,
            pspd=level2.pspd,
            section_marker=level2.section_marker,
            fftlen=level2.cfg.fftlen,
            sampling_freq=level2.cfg.sampling_freq,
            spatial_response_wavenum=level2.cfg.spatial_response_wavenum,
            freq_highpass=level2.cfg.freq_highpass,
            chunklen=level2.cfg.chunklen,
            chunkoverlap=level2.cfg.chunkoverlap,
        )

        return cls(
            Pk=Pk,
            k=k,
            Pf=Pf,
            freq=freq,
            platform_speed=platform_speed,
            cfg=level2.cfg,
        )

    @classmethod
    def from_atomix_netcdf(cls, fname: str):
        ds = xr.load_dataset(
            "MSS_BalticSea/MSS_Baltic.nc", group="L3_spectra"
        ).transpose(..., "N_SHEAR_SENSORS", "TIME_SPECTRA", "WAVENUMBER")

        return cls(
            Pk=ds["SH_SPEC"].values,
            k=ds["KCYC"].values,
            platform_speed=ds["PSPD_REL"].values,
            cfg=ShearConfig.from_atomix_netcdf(fname),
        )


def process_level3(
    shear: Float[ndarray, "n_shear time_fast"],
    pspd: Float[ndarray, "time_fast"],
    section_marker: Int[ndarray, "time_fast"],
    fftlen: int,
    sampling_freq: float,
    spatial_response_wavenum: float,
    freq_highpass: float,
    chunklen: int,
    chunkoverlap: int,
    ancillary: Dict[str, Float[ndarray, "time_fast"]] = None,  # average to time_slow
) -> Tuple[
    Float[ndarray, "time_slow k"],  # k
    Float[ndarray, "n_shear time_slow wavenumber"],  # Pk
    Float[ndarray, "n_shear time_slow wavenumber"],  # Pf
    Float[ndarray, "wavenumber"],  # freq
    Float[ndarray, "time_slow"],  # pspda
    Dict[str, Float[ndarray, "time_slow"]],  # ancillary_out
]:
    # segments = split_data(shear, section_marker)
    # for marker, data in segments.items(): # TODO

    Pf, freq = spectra(shear, fftlen, sampling_freq, chunklen, chunkoverlap)

    # Average to time_slow
    data_fast: Float[ndarray, "variable time_fast"] = (
        pspd[newaxis, :]  # add dimension
        if ancillary is None
        else np.stack((pspd, *(arr for k, arr in ancillary.items())), axis=0)
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
    correction_factor_spatial = apply_compensation_spatial_response(
        Pk, k, spatial_response_wavenum
    )
    _ = apply_compensation_highpass(Pk, freq, freq_highpass)
    # apply_removal_coherent_vibrations(P)
    # get_uncertainty_estimates(P)

    print(correction_factor_spatial)
    # raise ValueError(correction_factor_spatial)

    ancillary_out = (
        {
            name: (["time_slow"], data_slow[ind + 1, :])
            for ind, name in enumerate(ancillary.keys())
        }
        if ancillary is not None
        else {}
    )

    return k, Pk, Pf, freq, pspda, ancillary_out


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
) -> Float[ndarray, "time_slow k"]:
    correction_factor = 1.0 + (k / k0) ** 2
    # TODO Eqn. 18 text: Do not use spectra where correction exceeds 10
    correction_factor[correction_factor > 10.0] = 10.0  # dirty hack
    x *= correction_factor[newaxis, :, :]
    return correction_factor


def apply_compensation_highpass(
    x: Float[ndarray, "n_shear time_slow f"],
    freq: Float[ndarray, "f"],
    freq_highpass: float,
) -> Float[ndarray, "f"]:
    correction_factor = (1.0 + (freq_highpass / freq) ** 2.0) ** 2.0
    x /= correction_factor[newaxis, :]
    return correction_factor
