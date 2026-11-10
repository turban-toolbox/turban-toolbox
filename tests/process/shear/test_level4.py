import numpy as np
import pytest
import xarray as xr

from turban import ShearLevel3, ShearLevel4, ShearProcessing
from turban.process.shear.level4 import figure_of_merit
from turban.utils.filepaths import (
    atomix_benchmark_baltic_fpath,
    atomix_benchmark_faroe_fpath,
)

atomix_benchmark_fpaths = pytest.mark.parametrize(
    "fpath", [atomix_benchmark_baltic_fpath, atomix_benchmark_faroe_fpath]
)


def _fom_from_atomix_file(fpath):
    """FOM from the file's L3 spectra and L4 eps, KMAX and KVISC."""
    level3 = ShearLevel3.from_atomix_netcdf(fpath)
    ds4 = xr.load_dataset(fpath, group="L4_dissipation")
    fom, _, _ = figure_of_merit(
        waveno=level3.waveno,
        waveno_cutoff=ds4["KMAX"].values,
        eps=ds4["EPSI"].values,
        molvisc=ds4["KVISC"].values,
        psi=level3.psi_k_sh,
        log_psi_std=level3.cfg.log_psi_std,
    )
    return fom, ds4


def test_fom_from_level3():
    p = ShearProcessing.from_atomix_netcdf(atomix_benchmark_baltic_fpath, level=3)
    level4 = p.level4
    assert level4.fom.shape == level4.eps.shape
    assert np.all(level4.fom[np.isfinite(level4.fom)] > 0)

    # consistent with the kernel, recomputed from level 4 and level 3 fields
    fom, _, _ = figure_of_merit(
        waveno=p.level3.waveno,
        waveno_cutoff=level4.waveno_cutoff,
        eps=level4.eps,
        molvisc=level4.molvisc,
        psi=p.level3.psi_k_sh,
        log_psi_std=p.level3.cfg.log_psi_std,
    )
    np.testing.assert_allclose(level4.fom, fom)

    # exported and parsed back
    ds = level4.to_xarray()
    assert ds["fom"].dims == ("nshear", "time")
    np.testing.assert_allclose(ShearLevel4.from_xarray(ds).fom, level4.fom)


@atomix_benchmark_fpaths
def test_fom_mad_from_atomix_netcdf(fpath):
    # FOM present in the ATOMIX file is taken over, MAD recomputed by turban agrees
    level4 = ShearLevel4.from_atomix_netcdf(fpath)
    ds4 = xr.load_dataset(fpath, group="L4_dissipation")
    np.testing.assert_allclose(level4.fom, ds4["FOM"].values)
    valid = np.isfinite(ds4["MAD"].values) & np.isfinite(level4.log_diss_mad)
    assert valid.mean() > 0.9
    np.testing.assert_allclose(
        level4.log_diss_mad[valid], ds4["MAD"].values[valid], atol=0.02
    )


@atomix_benchmark_fpaths
def test_log_psi_std_from_atomix_netcdf(fpath):
    # N_V is read from the file, so sigma_ln_psi (Eq. 17) matches SPEC_STD
    level3 = ShearLevel3.from_atomix_netcdf(fpath)
    ds3 = xr.load_dataset(fpath, group="L3_spectra")
    assert level3.cfg.number_signals_vibration_removal == int(
        ds3["N_VIB_SENSORS"].values.item()
    )
    assert level3.cfg.number_signals_vibration_removal > 0
    np.testing.assert_allclose(
        level3.cfg.log_psi_std, ds3["SPEC_STD"].values.item(), rtol=1e-6
    )


@atomix_benchmark_fpaths
def test_fom_benchmark_against_atomix_netcdf(fpath):
    # FOM (Eq. 22) recomputed by turban from the file's spectra and dissipation
    # rates agrees with FOM stored in the file
    fom, ds4 = _fom_from_atomix_file(fpath)
    valid = np.isfinite(ds4["FOM"].values) & np.isfinite(fom)
    assert valid.mean() > 0.9
    np.testing.assert_allclose(fom[valid], ds4["FOM"].values[valid], atol=0.02)


@pytest.mark.parametrize(
    "pres, fom_ref",
    [
        # Table 3 in the ATOMIX paper (FOM_1, FOM_2 for spectral integration)
        (151.7, [0.66, 0.52]),
        (385.4, [0.57, 0.57]),
        (680.0, [0.64, 0.64]),
        (99.9, [2.02, 2.20]),
    ],
)
def test_fom_benchmark_faroe_table3(pres, fom_ref):
    fom, ds4 = _fom_from_atomix_file(atomix_benchmark_faroe_fpath)
    i = np.argmin(np.abs(ds4["PRES"].values - pres))
    assert abs(ds4["PRES"].values[i] - pres) < 0.1
    np.testing.assert_allclose(fom[:, i], fom_ref, atol=0.01)
