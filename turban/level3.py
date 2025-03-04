from dataclasses import dataclass

from beartype.typing import Tuple, Dict

from numpy import ndarray, newaxis
import numpy as np
from jaxtyping import Float, Int
import xarray as xr
from netCDF4 import Dataset

from turban.util import (
    reshape_overlap_index,
    reshape_any_nextlast,
    reshape_halfoverlap_last,
    average_fast_to_slow,
)
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
            fft_length=level2.cfg.fft_length,
            sampling_freq=level2.cfg.sampling_freq,
            spatial_response_wavenum=level2.cfg.spatial_response_wavenum,
            freq_highpass=level2.cfg.freq_highpass,
            fft_overlap=level2.cfg.fft_overlap,
            diss_length=level2.cfg.diss_length,
            diss_overlap=level2.cfg.diss_overlap,
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

    def to_xarray(self):
        return xr.Dataset(
            {
                "k": (["time_slow", "wavenumber"], self.k),
                "Pk": (["nshear", "time_slow", "wavenumber"], self.Pk),
                "Pf": (["nshear", "time_slow", "wavenumber"], self.Pf),
                "freq": (["wavenumber"], self.freq),
                "platform_speed": (["time_slow"], self.platform_speed),
            }
        )


def process_level3(
    shear: Float[ndarray, "n_shear time_fast"],
    pspd: Float[ndarray, "time_fast"],
    fft_length: int,
    fft_overlap: int,
    diss_length: int,
    diss_overlap: int,
    sampling_freq: float,
    spatial_response_wavenum: float,
    freq_highpass: float,
    section_marker: Int[ndarray, "time_fast"],
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
    ii = fast_to_slow_reshape_index(
        shear.shape[-1],
        fft_length,
        fft_overlap,
        diss_length,
        diss_overlap,
        section_marker,
    )

    Pf, freq = spectra(shear, sampling_freq, reshape_index=ii)

    # Average to time_slow
    data_fast: Float[ndarray, "variable time_fast"] = (
        pspd[newaxis, :]  # add dimension
        if ancillary is None
        else np.stack((pspd, *(arr for k, arr in ancillary.items())), axis=0)
    )
    # platform speed
    data_slow: Float[ndarray, "variable time_slow"] = average_fast_to_slow(
        data_fast, reshape_index=ii
    )
    pspda = data_slow[0, :]

    # to wavenumber domain
    Pk = Pf * pspda[newaxis, :, newaxis] / fft_length / (sampling_freq / 2)
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
    sampling_freq: float,
    fft_length: int = None,
    fft_overlap: int = None,
    diss_length: int = None,
    diss_overlap: int = None,
    section_marker: Int[ndarray, "time_fast"] = None,
    reshape_index: Int[ndarray, "diss_chunk fft_chunk fft_length"] = None,
) -> Tuple[
    Float[ndarray, "n_shear chunk freq"],
    Float[ndarray, "freq"],  # frequencies
]:
    """
    Produce spectra from cleaned shear time series.
    If reshape_index is not supplied, calculates it.
    """
    if reshape_index is None:
        reshape_index = fast_to_slow_reshape_index(
            shear.shape[-1],
            fft_length,
            fft_overlap,
            diss_length,
            diss_overlap,
            section_marker,
        )
    else:
        fft_length = reshape_index.shape[-1]

    # reshape to fft length windows
    yr = shear[..., reshape_index]
    # subtract mean
    yr -= yr.mean(axis=-1)[..., newaxis]
    # hanning window
    yr *= np.hanning(fft_length)[newaxis, newaxis, :]

    # periodograms
    freq = np.fft.rfftfreq(fft_length, d=1 / sampling_freq)
    Fyr = np.fft.rfft(yr)[:, :]
    Pf = (Fyr.conj() * Fyr).real
    # average spectra by chunks (reshape the segments)
    Pf = Pf.mean(axis=-2)
    return Pf, freq


def fast_to_slow_reshape_index(
    N: int,
    fft_length: int,
    fft_overlap: int,
    diss_length: int,
    diss_overlap: int,
    section_marker: Int[ndarray, "time_fast"] = None,
) -> Int[ndarray, "diss_chunk fft_chunk fft_length"]:

    if section_marker is None:
        section_marker = np.ones(N, dtype=int)

    sections = split_data(np.arange(N), section_marker)

    reshape_segments = []
    for data in sections.values():

        # reshape time dimension into chunks of length diss_length
        ii_diss: Int[ndarray, "diss_chunk diss_length"] = reshape_overlap_index(
            diss_length, diss_overlap, len(data)
        )
        # reshape fft dimension into chunks of length fft_length
        ii_fft: Int[ndarray, "fft_chunk fft_length"] = reshape_overlap_index(
            fft_length, fft_overlap, ii_diss.shape[-1]
        )
        reshape_segments.append(ii_diss[:, ii_fft] + data[0])

    # concatenate along dissipation chunks
    return np.concatenate(reshape_segments, axis=0)


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
