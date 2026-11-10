import numpy as np
from jaxtyping import Float
from numpy import ndarray

ArrayOrFloat = Float[ndarray, "..."] | float


def viscosity_kinematic(temp: ArrayOrFloat) -> ArrayOrFloat:
    """Kinematic viscosity of seawater at salinity 35.

    Parameters
    ----------
    temp : ndarray or float
        Temperature in degrees Celsius.

    Returns
    -------
    ndarray or float
        Kinematic viscosity in m2 s-1.
    """
    pol = [
        -1.131311019739306e-011,
        1.199552027472192e-009,
        -5.864346822839289e-008,
        1.828297985908266e-006,
    ]
    return np.polyval(pol, temp)


def _thermal_conductivity(temp: ArrayOrFloat, sal: ArrayOrFloat) -> ArrayOrFloat:
    """Thermal conductivity [W m-1 K-1] (Sharqawy et al. 2010, eq. 13)."""
    t68 = 1.00024 * np.asarray(temp)
    sp = np.asarray(sal) / 1.00472
    tk = t68 + 273.15
    return 10 ** (
        np.log10(240 + 0.0002 * sp)
        + 0.434
        * (2.3 - (343.5 + 0.037 * sp) / tk)
        * (1 - tk / (647.3 + 0.03 * sp)) ** (1 / 3)
        - 3
    )


def _specific_heat(temp: ArrayOrFloat, sal: ArrayOrFloat) -> ArrayOrFloat:
    """Specific heat capacity [J kg-1 K-1] (Sharqawy et al. 2010, eq. 9)."""
    t68 = 1.00024 * np.asarray(temp)
    sp = np.asarray(sal) / 1.00472
    a = 4206.8 - 6.6197 * sp + 1.2288e-2 * sp**2
    b = -1.1262 + 5.4178e-2 * sp - 2.2719e-4 * sp**2
    c = 1.2026e-2 - 5.3566e-4 * sp + 1.8906e-6 * sp**2
    d = 6.8777e-7 + 1.517e-6 * sp - 4.4268e-9 * sp**2
    return a + b * t68 + c * t68**2 + d * t68**3


def _density(temp: ArrayOrFloat, sal: ArrayOrFloat) -> ArrayOrFloat:
    """Density [kg m-3] at atmospheric pressure (Sharqawy et al. 2010, eq. 8)."""
    t = np.asarray(temp)
    s = np.asarray(sal) / 1000
    rho_w = (
        9.9992293295e02
        + 2.0341179217e-02 * t
        - 6.1624591598e-03 * t**2
        + 2.2614664708e-05 * t**3
        - 4.6570659168e-08 * t**4
    )
    d_rho = (
        8.0200240891e02 * s
        - 2.0005183488e00 * s * t
        + 1.6771024982e-02 * s * t**2
        - 3.0600536746e-05 * s * t**3
        - 1.6132224742e-05 * s**2 * t**2
    )
    return rho_w + d_rho


def viscosity_kinematic_full(temp: ArrayOrFloat, sal: ArrayOrFloat) -> ArrayOrFloat:
    """Kinematic viscosity of seawater at arbitrary salinity.

    Computed as nu = mu / rho at atmospheric pressure, both from Sharqawy,
    Lienhard & Zubair (2010). The dynamic viscosity correlation (based on
    Isdale et al. 1972 and El-Dessouky & Ettouney 2002) is::

        mu_sw = mu_w * (1 + A*S + B*S**2)
        A = 1.541 + 1.998e-2*T - 9.52e-5*T**2
        B = 7.974 - 7.561e-2*T + 4.724e-4*T**2
        mu_w = 4.2844e-5 + 1 / (0.157*(T + 64.993)**2 - 91.296)

    with T in degrees Celsius (ITS-90), S the reference salinity in kg/kg and
    ``mu_w`` the IAPWS 2008 pure water viscosity fit. Valid for
    0 <= T <= 180 degC and 0 <= S <= 150 g/kg with an accuracy of +-1.5 %.
    The pressure dependence (< 1 % over the oceanic range) is neglected.
    Unlike ``viscosity_kinematic`` this is not restricted to
    salinity 35.

    Parameters
    ----------
    temp : ndarray or float
        Temperature in degrees Celsius.
    sal : ndarray or float
        Practical salinity. Converted to reference salinity for the viscosity
        via S_R = (35.16504 / 35) * S_P.

    Returns
    -------
    ndarray or float
        Kinematic viscosity in m2 s-1.

    References
    ----------
    Sharqawy, M. H., Lienhard, J. H., & Zubair, S. M. (2010). Thermophysical
    properties of seawater: a review of existing correlations and data.
    Desalination and Water Treatment, 16, 354-380. doi:10.5004/dwt.2010.1079
    """
    t = np.asarray(temp)
    sal_ref = np.asarray(sal) * 35.16504 / 35
    s = sal_ref / 1000
    a = 1.541 + 1.998e-2 * t - 9.52e-5 * t**2
    b = 7.974 - 7.561e-2 * t + 4.724e-4 * t**2
    mu_w = 4.2844e-5 + 1 / (0.157 * (t + 64.993) ** 2 - 91.296)
    mu_sw = mu_w * (1 + a * s + b * s**2)
    return mu_sw / _density(temp, sal_ref)


def diffusivity_temp(temp: ArrayOrFloat, sal: ArrayOrFloat = 35.0) -> ArrayOrFloat:
    """Molecular thermal diffusivity of seawater.

    Parameters
    ----------
    temp : ndarray or float
        Temperature in degrees Celsius.
    sal : ndarray or float, optional
        Practical salinity. Default is 35.

    Returns
    -------
    ndarray or float
        Thermal diffusivity in m2 s-1.
    """
    return _thermal_conductivity(temp, sal) / (
        _density(temp, sal) * _specific_heat(temp, sal)
    )
