#!/bin/env python
from pathlib import Path

from numpy import ndarray
import xarray as xr
import numpy as np
from jaxtyping import Float

from turban.temperature.temperature import (
    diffusivity_temp,
    viscosity_kinematic,
)



def process(
    data: dict[str, Float[ndarray, "time"]],
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
        list[str] | None
    ) = None,  # fields from `data` to aggregate into level4
    level4_to_gradient: (
        list[str] | None
    ) = None,  # fields from `data` to aggregate into level4
    microtemp_to_average: (
        list[str] | None
    ) = None,  # fields from `data` to aggregate into microtemp
    microtemp_to_gradient: (
        list[str] | None
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


