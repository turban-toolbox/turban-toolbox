import numpy as np
import xarray as xr

from turban import ShearLevel4, ShearProcessing
from turban.process.shear.level4 import figure_of_merit
from turban.utils.filepaths import atomix_benchmark_baltic_fpath


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
        log_psi_var=p.level3.log_psi_var,
    )
    np.testing.assert_allclose(level4.fom, fom)

    # exported and parsed back
    ds = level4.to_xarray()
    assert ds["fom"].dims == ("nshear", "time")
    np.testing.assert_allclose(ShearLevel4.from_xarray(ds).fom, level4.fom)


def test_fom_from_atomix_netcdf():
    # FOM present in the ATOMIX file is taken over as is
    level4 = ShearLevel4.from_atomix_netcdf(atomix_benchmark_baltic_fpath)
    ref = xr.load_dataset(atomix_benchmark_baltic_fpath, group="L4_dissipation")
    np.testing.assert_allclose(level4.fom, ref["FOM"].values)
