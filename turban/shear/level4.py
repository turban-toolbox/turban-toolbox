from numpy import ndarray, newaxis
import numpy as np
from jaxtyping import Float

from turban.util import integrate

def process_level4(
    psi: Float[ndarray, "nshear time wavenumber"],
    wavenumber: Float[ndarray, "time wavenumber"],
    platform_speed: Float[ndarray, "time"],
    waveno_cutoff_spatial_corr: float,
    freq_cutoff_antialias: float,
    freq_cutoff_corrupt: float,
) -> tuple[
    Float[ndarray, "nshear time"],  # eps estimate
    Float[ndarray, "nshear time"],  # eps_specint
    Float[ndarray, "nshear time"],  # eps_isrfit
]:
    """
    Produce epsilon estimates from shear power spectra.
    """
    # nu = get_seawater_viscosity(999.)
    nu = np.array(1.6e-6)[newaxis]
    # set psi=0 at k=0 (see text just after Eq. 27)
    psi[:, :, 0] = 0.0

    # 1st estimate
    eps1 = get_eps_first_estimate(psi, wavenumber, nu)

    # Inertial subrange integration
    eps_specint = spectrum_integration(
        psi,
        wavenumber,
        eps1,
        nu,
        platform_speed,
        waveno_cutoff_spatial_corr,
        freq_cutoff_antialias,
        freq_cutoff_corrupt,
    )

    k_kolmogorov = (eps1 / nu**3) ** 0.25
    eps_isrfit = inertial_range_fit(psi, wavenumber, k_kolmogorov)

    eps = np.where(eps1 < 1e-5, eps_specint, eps_isrfit)

    return eps, eps_specint, eps_isrfit


def inertial_range_fit(
    psi: Float[ndarray, "nshear time wavenumber"],
    wavenumber: Float[ndarray, "time wavenumber"],
    k_kolmogorov: Float[ndarray, "nshear time"],
    a_kolmogorov: float = 8.19,
) -> Float[ndarray, "nshear time"]:
    """
    See Eq. 28. This assumes a known Kolmogorov constant, as opposed to linear
    regression to both slope and offset.

    The default value for A corresponds to Kolomgorov's constant for 1-dimensional
    spectra C1=0.53.

    TODO: find upper limit of inertial subrange (function of k_kolmogorov?)
    """
    ln_epsilon = 1.5 * (
        np.log(psi) - np.log(a_kolmogorov) - np.log(wavenumber[newaxis, ...]) / 3
    )

    ln_epsilon_fitrange = np.where(wavenumber[newaxis, ...] < 3, ln_epsilon, np.nan)
    return np.exp(np.nanmean(ln_epsilon_fitrange, axis=2))


def spectrum_integration(
    psi: Float[ndarray, "nshear time wavenumber"],
    wavenumber: Float[ndarray, "time wavenumber"],
    eps1: Float[ndarray, "nshear time"],
    nu: Float[ndarray, "time"],
    platform_speed: Float[ndarray, "time"],
    waveno_cutoff_spatial_corr: float,
    freq_cutoff_antialias: float,
    freq_cutoff_corrupt: float,
):
    # 2nd estimate
    (eps2, waveno_cutoff) = get_eps_second_estimate(
        psi,
        wavenumber,
        eps1,
        nu,
        platform_speed,
        waveno_cutoff_spatial_corr,
        freq_cutoff_antialias,
        freq_cutoff_corrupt,
    )

    # 3rd etc. estimates
    eps_increase = -999.0
    # start value of iteration convergence measure
    eps = eps2
    eps_previous = eps2
    while np.any(eps_increase > 1.01):
        eps_previous = eps
        eps /= get_spectral_variance_resolved_fraction(waveno_cutoff, eps, nu)
        eps_increase = eps / eps_previous

    return eps


def get_eps_first_estimate(
    psi: Float[ndarray, "nshear time wavenumber"],
    wavenumber: Float[ndarray, "time wavenumber"],
    nu: Float[ndarray, "time"],
) -> Float[ndarray, "nshear time_slow"]:  # eps estimate
    # integrate to 10 cpm
    eps10 = (
        7.5
        * nu
        * integrate(psi, wavenumber, np.array(0.0)[newaxis], np.array(10.0)[newaxis])
    )
    # Eq. 24
    a = 1.25e-9 * nu**3
    b = 5.5e-8 * nu ** (-2.5)
    eps = (np.sqrt(1.0 + a * eps10) + np.exp(-b * eps10) - 1.0) * eps10
    return eps


def get_eps_second_estimate(
    psi: Float[ndarray, "nshear time wavenumber"],
    wavenumber: Float[ndarray, "time wavenumber"],
    eps1: Float[ndarray, "nshear time"],
    nu: Float[ndarray, "time"],
    platform_speed: Float[ndarray, "time"],
    waveno_cutoff_spatial_corr: float,
    freq_cutoff_antialias: float,
    freq_cutoff_corrupt: float,
) -> tuple[
    Float[ndarray, "nshear time"],  # eps estimate
    Float[ndarray, "nshear time"],  # cutoff wavenumber
]:
    k95: Float[ndarray, "nshear time"] = 0.12 * (eps1 / nu**3) ** 0.25
    waveno_spectral_min = 9999.0  # TODO
    waveno_cutoff_antialias: Float[ndarray, "time"] = (
        0.9 * freq_cutoff_antialias / platform_speed
    )
    waveno_cutoff_corrupt: Float[ndarray, "time"] = freq_cutoff_corrupt / platform_speed
    ku: Float[ndarray, "nshear time"] = np.min(
        np.stack(
            np.broadcast_arrays(
                k95,
                np.array(waveno_spectral_min)[newaxis, newaxis],
                np.array(waveno_cutoff_spatial_corr)[newaxis, newaxis],
                np.array(waveno_cutoff_antialias)[newaxis, :],
                np.array(waveno_cutoff_corrupt)[newaxis, :],
            ),
            axis=-1,
        ),
        axis=-1,
    )
    return (
        7.5 * nu * integrate(psi, wavenumber, np.array(0.0)[newaxis, newaxis], ku),
        ku,
    )


def get_spectral_variance_resolved_fraction(
    waveno: Float[ndarray, "time"],
    eps: Float[ndarray, "nshear time"],
    nu: Float[ndarray, "time"],
) -> Float[ndarray, "nshear time"]:
    # Eq. 11; I_L (3rd model)
    length_kolmogorov = np.sqrt(nu[newaxis, :] ** 3 / eps)
    k43 = (waveno[newaxis, :] * length_kolmogorov) ** (4.0 / 3.0)
    return np.tanh(65.5 * k43) - 9.0 * k43 * np.exp((-54.5 * k43))
