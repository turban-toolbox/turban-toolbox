import numpy as np
import pytest

from turban.utils.seawater import (
    _density,
    _specific_heat,
    _thermal_conductivity,
    diffusivity_temp,
    viscosity_kinematic,
    viscosity_kinematic_full,
)


def test_viscosity_kinematic_simple_vs_full():
    # Polynomial fit vs. full Sharqawy et al. (2010) at S=35 over the oceanic
    # temperature range; they agree within the +-1.5 % accuracy of the
    # correlation (slightly more near the freezing point)
    temp = np.linspace(-2, 35, 75)
    nu_poly = viscosity_kinematic(temp)
    nu_full = viscosity_kinematic_full(temp, 35.0)
    assert nu_full.shape == temp.shape
    np.testing.assert_allclose(nu_poly, nu_full, rtol=0.02)
    np.testing.assert_allclose(nu_poly[temp >= 0], nu_full[temp >= 0], rtol=0.015)


def test_viscosity_kinematic_full_salinity_deviation():
    # nu increases with salinity by 0.11-0.16 % per psu (largest in warm water),
    # so for S=29..34 the deviation from S=35 stays below 1 %, i.e. within the
    # accuracy of the correlation
    temp = np.linspace(-2, 35, 75)[:, np.newaxis]
    sal = np.arange(29, 35, dtype=float)
    nu = viscosity_kinematic_full(temp, sal)
    nu_35 = viscosity_kinematic_full(temp, 35.0)
    rel_dev = nu / nu_35 - 1
    assert rel_dev.shape == (75, 6)
    assert np.all(np.diff(nu, axis=1) > 0)
    np.testing.assert_array_less(np.abs(rel_dev), 0.01)
    dev_per_psu = rel_dev / (sal - 35)
    assert np.all((dev_per_psu > 0.0011) & (dev_per_psu < 0.0017))
    # using the S=35 polynomial in fresher water stays within 2 %
    nu_poly = viscosity_kinematic(temp)
    np.testing.assert_allclose(nu_poly * np.ones_like(nu), nu, rtol=0.02)


def test_seawater_properties_reference_values():
    # Sharqawy et al. (2010) correlations at T=20 degC, S=35 vs. their tabulated
    # values (rho ~ 1024.8 kg m-3, cp ~ 3993 J kg-1 K-1, k ~ 0.60 W m-1 K-1)
    np.testing.assert_allclose(_density(20.0, 35.0), 1024.8, rtol=1e-3)
    np.testing.assert_allclose(_specific_heat(20.0, 35.0), 3993.0, rtol=5e-3)
    np.testing.assert_allclose(_thermal_conductivity(20.0, 35.0), 0.60, rtol=1e-2)


def test_diffusivity_temp():
    # thermal diffusivity of seawater is ~1.4-1.5e-7 m2 s-1 and increases with
    # temperature; default salinity is 35
    np.testing.assert_allclose(diffusivity_temp(20.0), 1.47e-7, rtol=1e-2)
    assert diffusivity_temp(20.0) == diffusivity_temp(20.0, 35.0)
    temp = np.linspace(-2, 30, 33)
    kappa = diffusivity_temp(temp)
    assert kappa.shape == temp.shape
    assert np.all(np.diff(kappa) > 0)
    assert np.all((kappa > 1.3e-7) & (kappa < 1.6e-7))

