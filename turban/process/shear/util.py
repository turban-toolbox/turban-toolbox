from numpy import ndarray, newaxis
from jaxtyping import Float

from turban.utils.util import kolmogorov_length


def psi_nondim_factor(
    eps: Float[ndarray, "*any"],
    molvisc: Float[ndarray, "*any"],
) -> Float[ndarray, "*any"]:
    """To pass from non-dimensional to dimensional shear spectra, see Eq. 6"""
    return (eps**3 / molvisc) ** 0.25


def model_spectrum(
    waveno: Float[ndarray, "*any waveno"],
    eps: Float[ndarray, "*any"],
    molvisc: Float[ndarray, "*any"],
) -> Float[ndarray, "*any waveno"]:
    """Uses the Lueck spectrum (Eq. 9) - consistent with
    `get_spectral_variance_resolved_fraction`"""
    k_nondim = waveno * kolmogorov_length(eps, molvisc)[..., newaxis]
    psi_nondim = model_spectrum_lueck(k_nondim)
    return psi_nondim * psi_nondim_factor(eps, molvisc)[..., newaxis]


def model_spectrum_lueck(
    waveno_nondim: Float[ndarray, "*any waveno"],  # nondimensional waveno
) -> Float[ndarray, "*any waveno"]:
    """Non-dimensional form, Eq. 9"""
    y = (waveno_nondim / 0.015) ** 2
    fac1 = 8.048 * waveno_nondim ** (1 / 3) / (1 + (21.7 * waveno_nondim) ** 3)
    fac2 = 1 / (1 + (6.6 * waveno_nondim) ** 2.5)
    fac3 = 1 + 0.36 * y / ((y - 1) ** 2 + 2 * y)
    return fac1 * fac2 * fac3
