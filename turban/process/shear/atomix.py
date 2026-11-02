"""Loaders for ATOMIX netCDF files (groups ``L1_converted``, ``L2_cleaned``,
``L3_spectra``, ``L4_dissipation``).

``config_attrs`` provides the ``ShearConfig`` attributes. ``levelN_kwarg`` provide
the constructor kwargs (without ``level_below``) for the ``ShearLevelN.from_atomix_netcdf``
classmethods in ``turban.process.shear.api``.
"""

import numpy as np
import xarray as xr
from jaxtyping import Float
from netCDF4 import Dataset
from numpy import newaxis, ndarray

from turban.process.generic.api import AuxDataTypehintLevel12
from turban.process.generic.config import SegmentConfig
from turban.process.shear.config import ShearConfig
from turban.process.shear.level4 import figure_of_merit
from turban.utils.logging import get_logger

logger = get_logger(__name__)

# ATOMIX variables holding the signals for vibration removal, in order of preference
_VIB_VARS = ("VIB", "ACC")


def _vib_aux_data(fname: str, groups: tuple[str, ...]) -> AuxDataTypehintLevel12:
    """Vibration signals (``VIB``, else ``ACC``) of the first of ``groups`` holding
    any, as level 1/2 aux data named ``VIB1, VIB2, ...`` (``ACC1, ...``), aggregated
    with ``mean``."""
    for group in groups:
        with xr.open_dataset(fname, group=group) as ds:
            for var in _VIB_VARS:
                if var in ds:
                    arr = ds[var].transpose(f"N_{var}_SENSORS", ...).values
                    return {
                        f"{var}{i}": (["time"], a, {"mean": None})
                        for i, a in enumerate(arr, start=1)
                    }
    return {}


def _warn_n_vib_mismatch(fname: str, n_vib: int):
    """Warn if ``N_VIB_SENSORS`` in ``L3_spectra`` (if present) differs from the
    number ``n_vib`` of vibration signals found."""
    with xr.open_dataset(fname, group="L3_spectra") as ds:
        n_vib_file = (
            int(ds["N_VIB_SENSORS"].values.item())
            if "N_VIB_SENSORS" in ds
            else ds.sizes.get("N_VIB_SENSORS")
        )
    if n_vib_file is not None and n_vib_file != n_vib:
        logger.warning(
            f"N_VIB_SENSORS={n_vib_file} in {fname}, but {n_vib} vibration signals "
            "found in L1_converted/L2_cleaned; log_psi_std uses the latter."
        )


def config_attrs(fname: str) -> dict:
    """``ShearConfig`` attributes from an ATOMIX netCDF file.

    ``vibration_channels`` are not set, as they require the corresponding aux data
    (see ``level1_kwarg``).
    """
    attrs = SegmentConfig._attrs_from_atomix_netcdf(fname)
    with Dataset(fname) as f:
        attrs["freq_highpass"] = f.HP_cut
    return attrs


def level1_kwarg(fname: str) -> dict:
    """``ShearLevel1`` kwargs from an ATOMIX netCDF file, including the config.

    The vibration signals of ``L1_converted``, else ``L2_cleaned``, are added as aux
    data, and their names as ``cfg.vibration_channels``.
    """
    vib = _vib_aux_data(fname, ("L1_converted", "L2_cleaned"))
    _warn_n_vib_mismatch(fname, len(vib))
    # TODO: handle section_number through level 2
    with (
        xr.open_dataset(fname, group="L1_converted") as ds,
        xr.open_dataset(fname, group="L2_cleaned") as ds2,
    ):
        return dict(
            time=ds["TIME"].values,
            senspeed=(ds if "PSPD_REL" in ds else ds2)["PSPD_REL"].values,
            shear=ds["SHEAR"].values,
            section_number=ds2["SECTION_NUMBER"].values.astype(int),
            cfg=ShearConfig(**config_attrs(fname), vibration_channels=tuple(vib)),
            _aux_data=vib,
        )


def level2_kwarg(fname: str) -> dict:
    """``ShearLevel2`` kwargs from an ATOMIX netCDF file, including the vibration
    signals of ``L2_cleaned`` as aux data, named as ``cfg.vibration_channels`` of
    level 1. Without these, those of level 1 are used, as level 2 takes over the aux
    data of level 1."""
    with xr.open_dataset(fname, group="L2_cleaned") as ds:
        shear = ds["SHEAR"].values
        return dict(
            time=ds["TIME"].values,
            shear=shear,
            senspeed=ds["PSPD_REL"].values,
            section_number=ds["SECTION_NUMBER"].values.astype(int),
            # TODO: this does not seem to be saved in (all?) ATOMIX files
            num_despike_iter=9999 * np.zeros_like(shear, dtype=int),
            _aux_data=_vib_aux_data(fname, ("L2_cleaned",)),
        )


def level3_kwarg(fname: str) -> dict:
    """``ShearLevel3`` kwargs from an ATOMIX netCDF file."""
    with xr.open_dataset(fname, group="L3_spectra") as ds:
        ds = ds.transpose(
            ...,
            "N_SHEAR_SENSORS",
            "TIME_SPECTRA",
            "WAVENUMBER" if "WAVENUMBER" in ds.dims else "N_WAVENUMBER",
        )
        # spectra after vibration-coherent noise removal (Goodman), if available
        spec_var = "SH_SPEC_CLEAN" if "SH_SPEC_CLEAN" in ds else "SH_SPEC"
        logger.debug(f"Loading shear spectra from {spec_var} in {fname}")
        psi_k_sh = ds[spec_var].values
        return dict(
            time=ds["TIME"].values,
            psi_k_sh=psi_k_sh,
            waveno=ds["KCYC"].values,
            psi_f_sh=psi_k_sh * np.nan,
            freq=np.nan * np.ones(ds["KCYC"].values.shape[-1]),
            senspeed=ds["PSPD_REL"].values,
            section_number=ds["SECTION_NUMBER"].values.astype(int),
            spike_fraction=np.nan * np.ones_like(psi_k_sh[:, :, 0]),
            # TODO: this does not seem to be saved in (all?) ATOMIX files
            max_despike_iter=9999 * np.ones_like(psi_k_sh[:, :, 0], dtype=int),
        )


def level4_kwarg(
    fname: str,
    waveno: Float[ndarray, "time waveno"],
    psi_k_sh: Float[ndarray, "nshear time waveno"],
    log_psi_std: float,
) -> dict:
    """``ShearLevel4`` kwargs from an ATOMIX netCDF file. The level 3 ``waveno``,
    ``psi_k_sh`` and ``log_psi_std`` are needed for the MAD (and the FOM, if not in
    the file)."""
    with xr.open_dataset(fname, group="L4_dissipation") as ds:
        eps = ds["EPSI"].values
        molvisc = ds["KVISC"].values
        waveno_cutoff = ds["KMAX"].values
        fom, log_diss_mad, _ = figure_of_merit(
            waveno=waveno,
            waveno_cutoff=waveno_cutoff,
            eps=eps,
            molvisc=molvisc,
            psi=psi_k_sh,
            log_psi_std=log_psi_std,
        )
        return dict(
            eps=eps,
            time=ds["TIME"].values,
            eps_source_flag=ds["METHOD"].values.astype(int) + 1,
            section_number=ds["SECTION_NUMBER"].values.astype(int),
            quality_metric=ds["EPSI_FLAGS"].values.astype(int),
            molvisc=molvisc,
            resolved_var_frac=ds["VAR_RESOLVED"].values,
            fom=ds["FOM"].values if "FOM" in ds else fom,
            num_spec_points=ds["N_S"].values.astype(int),
            waveno_cutoff=waveno_cutoff,
            kolmlen=(molvisc[newaxis, :] ** 3 / eps) ** 0.25,
            log_diss_std=(
                ds["EPSI_STD"].values
                if "EPSI_STD" in ds
                else np.nan * np.ones_like(eps)
            ),
            log_diss_mad=log_diss_mad,
        )
