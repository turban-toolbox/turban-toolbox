#!/bin/env python
from pathlib import Path

from beartype.typing import Dict, Tuple, List
from numpy import ndarray
import xarray as xr
import numpy as np
from jaxtyping import Float
from netCDF4 import Dataset

from turban.util import average_fast_to_slow, binned_gradient_halfoverlap
from turban.level1 import get_vsink, fft_grad, ShearLevel1
from turban.level2 import ShearLevel2, select_sections
from turban.level3 import ShearLevel3
from turban.level4 import ShearLevel4
from turban.temperature import (
    temperature_dissipation,
    diffusivity_temp,
    viscosity_kinematic,
)


def fast_to_slow_grad_by_segment(
    x: Float[ndarray, "... time_fast"],
    pspd: Float[ndarray, "... time_fast"],
    section_select_idx: List[List[int]],
    fft_length: int,
    sampling_freq: float,
) -> Float[ndarray, "... time_slow"]:
    """
    Calculate the gradient of `x` with respect to depth, averaged over each segment.
    This is done by using pspd, the platform speed, to convert between time and depth.
    """
    x_segments = [x[..., inds] for inds in section_select_idx]
    pspd_segments = [pspd[..., inds] for inds in section_select_idx]
    dxdz_segments = []
    for x_seg, pspd_seg in zip(x_segments, pspd_segments):
        # TODO: 3 * fft_length is valid for chunklen=5, chunkoverlap=2
        dxdz_segments.append(
            binned_gradient_halfoverlap(x_seg, pspd_seg, 3 * fft_length, sampling_freq)
        )
    return np.concatenate([xx for xx in dxdz_segments], axis=-1)


def fast_to_slow_avg_by_segment():
    pass


def load(fname):
    with Dataset(fname) as f:
        groups = list(f.groups)

    if "level0" in groups:
        ds0 = xr.load_dataset(fname, group="level0")
    else:
        ds0 = None

    if "microtemp" in groups:
        dst = xr.load_dataset(fname, group="microtemp")
    else:
        dst = None

    if "level3" in groups:
        ds3 = xr.load_dataset(fname, group="level3")
    else:
        ds3 = None

    if "level4" in groups:
        ds4 = xr.load_dataset(fname, group="level4")
    else:
        ds4 = None

    return ds0, ds3, ds4, dst


def process(
    data: Dict[str, Float[ndarray, "time"]],
    sampling_freq: float,
    fft_length_microtemp: int,
    fft_length_shear: int,
    chunklen_microtemp: int = 5,
    chunklen_shear: int = 5,
    chunkoverlap_microtemp: int = 2,
    chunkoverlap_shear: int = 2,
    outfile: str | None = None,
    pspd_acceptable_minimum: float = 0.3,  # minimum acceptable platform speed
    level4_to_average: (
        List[str] | None
    ) = None,  # fields from `data` to aggregate into level4
    level4_to_gradient: (
        List[str] | None
    ) = None,  # fields from `data` to aggregate into level4
    microtemp_to_average: (
        List[str] | None
    ) = None,  # fields from `data` to aggregate into microtemp
    microtemp_to_gradient: (
        List[str] | None
    ) = None,  # fields from `data` to aggregate into microtemp
):
    if level4_to_average is None:
        level4_to_average = []
    if level4_to_gradient is None:
        level4_to_gradient = []
    if microtemp_to_average is None:
        microtemp_to_average = []
    if microtemp_to_gradient is None:
        microtemp_to_gradient = []

    if Path(outfile).exists():
        Path(outfile).unlink(missing_ok=True)

    data["pspd"], data["pressure_lp"] = get_vsink(data["PRESSURE"], sampling_freq)
    pspd = data["pspd"]
    pspd_potentially_valid = pspd[pspd > pspd_acceptable_minimum]
    pspda = np.nanmedian(pspd_potentially_valid)

    ds0 = xr.Dataset({k: (["time_fast"], v) for k, v in data.items()})
    ds0.to_netcdf(outfile, mode="a", group="level0")
    # Temperature microstructure

    section_select_criteria = [
        ((12.0, None), data["pressure_lp"]),
        ((0.9 * pspda, 1.1 * pspda), data["pspd"]),
    ]

    section_select_idx = select_sections(
        section_select_criteria, segment_min_len=fft_length_microtemp * 3
    )

    chi, k_batchelor = microtemp(
        data["TEMP_EMPH"],
        data["pspd"],
        section_select_idx,
        sampling_freq=sampling_freq,
        fft_length=fft_length_microtemp,
        chunklen=5,
        chunkoverlap=2,
        outfile=outfile,
    )

    dst = xr.load_dataset(outfile, group="microtemp")

    microtemp_averaged, microtemp_gradients = aggregate(
        [data["pressure_lp"]] + [data[v] for v in microtemp_to_average],
        ([data["TEMP"], data["SIGMA0"]] + [data[v] for v in microtemp_to_gradient]),
        data["pspd"],
        section_select_idx,
        sampling_freq,
        fft_length_microtemp,
    )
    pressure = microtemp_averaged[0]

    dst["PRESSURE"] = (["time_slow"], pressure)

    dTdz_mean, drhodz = microtemp_gradients[:2]

    for i, v in enumerate(microtemp_to_average[1:]):
        dst[v] = (["time_slow"], microtemp_averaged[i + 1])
    for i, v in enumerate(microtemp_to_gradient[2:]):
        dst[v + "_gradient"] = (["time_slow"], microtemp_gradients[+2])

    dst["dTdz"] = (["time_slow"], dTdz_mean)
    dst["N2"] = (["time_slow"], drhodz * 9.81 / np.nanmean(data["SIGMA0"]))

    varTz = dst["chi"] / 6 / diffusivity_temp
    dst["Cx"] = varTz / dst["dTdz"] ** 2
    dst["Kt"] = 3 * diffusivity_temp * dst["Cx"]
    dst["eps"] = (
        dst["k_batchelor_estimate"] ** (4.0)
        * viscosity_kinematic
        * diffusivity_temp ** (2.0)
    )
    dst["lozm"] = np.sqrt(dst["eps"] / dst["N2"] ** 1.5)

    dst["FH"] = dst["Kt"] * dst["dTdz"] * 4e6
    dst["Reb"] = dst["eps"] / viscosity_kinematic / dst["N2"]

    dst.to_netcdf(outfile, mode="a", group="microtemp")

    # Shear microstructure

    section_select_criteria = [
        ((5.0, None), data["pressure_lp"]),
        ((0.9 * pspda, 1.1 * pspda), data["pspd"]),
        ((-10.0, 10.0), data["pitch"]),
        ((-10.0, 10.0), data["roll"]),
    ]

    section_select_idx = select_sections(
        section_select_criteria, segment_min_len=fft_length_shear * 3
    )

    wavenumber, spectra, epsilon = shear(
        np.array([data[k] for k in ["SHEAR_1", "SHEAR_2"]]),
        data["pspd"],
        section_select_idx,
        fft_length=fft_length_shear,
        sampling_freq=sampling_freq,
        chunklen=chunklen_shear,
        chunkoverlap=chunkoverlap_shear,
        outfile=outfile,
    )

    ds3 = xr.load_dataset(outfile, group="level3")
    ds4 = xr.load_dataset(outfile, group="level4")

    level4_averaged, level4_gradients = aggregate(
        [data["pressure_lp"]] + [data[v] for v in level4_to_average],
        [data[v] for v in level4_to_gradient],
        None,
        section_select_idx,
        sampling_freq,
        fft_length_shear,
    )
    pressure = level4_averaged[0]

    for i, v in enumerate(level4_to_average[1:]):
        dst[v] = (["time_slow"], level4_averaged[i + 1])
    for i, v in enumerate(level4_to_gradient):
        dst[v + "_gradient"] = (["time_slow"], level4_gradients)

    ds4["PRESSURE"] = (["time_slow"], pressure)

    ds4.to_netcdf(outfile, mode="a", group="level4")

    return ds0, ds3, ds4, dst


def aggregate(
    to_average: List[Float[ndarray, "time_fast"]],
    to_gradient: List[Float[ndarray, "time_fast"]],
    pspd: Float[ndarray, "time_fast"] | None,
    section_select_idx: List[List[int]],
    sampling_freq: float,
    fft_length: int,
    chunklen: int = 5,
    chunkoverlap: int = 2,
) -> Tuple[
    List[Float[ndarray, "time_slow"]],
    List[Float[ndarray, "time_slow"]],
]:
    average = []
    if len(to_average) > 0:
        for x in to_average:
            segments = [x[..., inds] for inds in section_select_idx]
            segments_out = []
            for segment in segments:
                segments_out.append(
                    average_fast_to_slow(
                        segment, fft_length, chunklen=chunklen, chunkoverlap=chunkoverlap
                    )
                )
            average.append(np.concatenate(segments_out, axis=-1))

    gradient = []
    if len(to_gradient) > 0:
        pspd_segments = [pspd[..., inds] for inds in section_select_idx]
        for x in to_gradient:
            segments = [x[..., inds] for inds in section_select_idx]
            segments_out = []
            for segment, pspd_segment in zip(segments, pspd_segments):
                segments_out.append(
                    binned_gradient_halfoverlap(
                        segment, pspd_segment, 3 * fft_length, sampling_freq
                    )
                )
            gradient.append(np.concatenate(segments_out, axis=-1))

    return average, gradient


def microtemp(
    temp_emph: Float[ndarray, "time_fast"],
    pspd: Float[ndarray, "time_fast"],
    section_select_idx: List[List[int]],
    sampling_freq: float,
    fft_length: int,
    chunklen: int = 5,
    chunkoverlap: int = 2,
    outfile: str | None = None,
) -> Tuple[Float[ndarray, "... time_slow"], ...]:
    """
    Process temperature microstructure.

    Default values to calculate spectra using 3+2 half-overlapping FFT
    intervals, i.e. 3*2048 samples.
    """

    dTdt = fft_grad(temp_emph, 1 / sampling_freq)

    dTdt_segments = [dTdt[..., inds] for inds in section_select_idx]
    pspd_segments = [pspd[..., inds] for inds in section_select_idx]

    out = []

    for dTdt_segment, pspd_segment in zip(dTdt_segments, pspd_segments):
        (
            k,
            Pk,
            chi,
            k_batchelor_estimate,
        ) = temperature_dissipation(
            dTdt=dTdt_segment,
            pspd=pspd_segment,
            chunklen=chunklen,
            chunkoverlap=chunkoverlap,
            fft_length=fft_length,
            sampling_freq=sampling_freq,
        )

        out.append(
            xr.Dataset(
                data_vars={
                    "k": (["time_slow", "wavenumber"], k),
                    "Pk": (["time_slow", "wavenumber"], Pk),
                    # "Pnoise": (["time_slow", "wavenumber"], Pnoise),
                    "chi": (["time_slow"], chi),
                    "k_batchelor_estimate": (["time_slow"], k_batchelor_estimate),
                }
            )
        )

    ds = xr.concat(out, dim="time_slow")
    if outfile is not None:
        ds.to_netcdf(outfile, mode="a", group="microtemp")

    return (
        ds["chi"].values,
        ds["k_batchelor_estimate"].values,
    )


def shear(
    shear: Float[ndarray, "n_shear time_fast"],
    pspd: Float[ndarray, "time_fast"],
    section_select_idx: List[List[int]],
    fft_length: int,  # length of single FFT segment to estimate periodogram
    sampling_freq: float,  # Hz
    spatial_response_wavenum: float = 50.0,
    freq_highpass: float = 0.15,
    # length of data segment to estimate dissipation. default: 2.5 x FFT segment
    chunklen: int = 5,
    # length of overlap between consecutive dissipation segments
    chunkoverlap: int = 2,
    outfile: str | None = None,
) -> Tuple[
    Float[ndarray, "time_slow wavenum"],
    # spectra
    Float[ndarray, "nshear time_slow wavenum"],
    # dissipation
    Float[ndarray, "nshear time_slow"],
]:

    raise NotImplementedError("Deprecated?")
    shear = process_level1(pspd, shear)

    shear_segments_cleaned = process_level2(
        shear, section_select_idx, sampling_freq, fft_length
    )
    pspd_segments = [pspd[..., inds] for inds in section_select_idx]

    nshear = shear.shape[0]
    nfreq = int(fft_length / 2 + 1)
    ks = np.zeros((0, nfreq))
    Pks = np.zeros((nshear, 0, nfreq))
    epsilons = np.zeros((nshear, 0))

    out3 = []
    out4 = []
    for (shear_segment, _, _), pspd_segment in zip(
        shear_segments_cleaned, pspd_segments
    ):

        pspda = average_fast_to_slow(
            pspd_segment, fft_length, chunklen=chunklen, chunkoverlap=chunkoverlap
        )

        ds3 = process_level3(
            shear_segment,
            pspd_segment,
            fft_length=fft_length,
            sampling_freq=sampling_freq,
            spatial_response_wavenum=spatial_response_wavenum,
            freq_highpass=freq_highpass,
            chunklen=chunklen,
            chunkoverlap=chunkoverlap,
        )
        ds4 = process_level4(ds3["Pk"].values, ds3["k"].values, pspda)

        out3.append(ds3)
        out4.append(ds4)

    ds3 = xr.concat(out3, dim="time_slow")
    ds4 = xr.concat(out4, dim="time_slow")

    if outfile is not None:
        ds3.to_netcdf(outfile, mode="a", group="level3")
        ds4.to_netcdf(outfile, mode="a", group="level4")

    return ds3["k"].values, ds3["Pk"].values, ds4["eps"].values
