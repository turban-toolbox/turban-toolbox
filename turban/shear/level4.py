from multiprocessing import Value
from selectors import EpollSelector
from unicodedata import numeric
from numpy import ndarray, newaxis
import numpy as np
from jaxtyping import Float, Int

from turban.util import integrate


def process_level4(
    psi: Float[ndarray, "nshear time wavenumber"],
    wavenumber: Float[ndarray, "time wavenumber"],
    platform_speed: Float[ndarray, "time"],
    waveno_cutoff_spatial_corr: float,
    freq_cutoff_antialias: float,
    freq_cutoff_corrupt: float,
    data_length: Float[ndarray, "time"],
    log_psi_var: float,
) -> tuple[
    Float[ndarray, "nshear time"],  # eps estimate
    Int[ndarray, "nshear time"],  # eps_source_flag. 1: spec_int, 2: isr_fit
    Float[ndarray, "nshear time"],  # log(eps) variance
    Float[ndarray, "nshear time"],  # kolmogorov length as per Eq. 29
    Float[ndarray, "nshear time"],  # resolved fraction of shear variance
    Float[ndarray, "nshear time"],  # Figure of Merit
    Float[ndarray, "nshear time"],  # Mean Absolute Deviation of log(psi)
    Int[ndarray, "nshear time"],  # number of spectral points
]:
    """
    Produce epsilon estimates from shear power spectra.
    """
    eps_crit = 1e-5
    # nu = get_seawater_viscosity(999.)
    mol_visc = np.array(1.6e-6)[
        newaxis
    ]  # TODO: get from temperature (aggregate in level3)
    # set psi=0 at k=0 (see text just after Eq. 27)
    psi[:, :, 0] = 0.0

    # 1st estimate
    eps1 = get_eps_first_estimate(psi, wavenumber, mol_visc)

    # Inertial subrange integration
    eps_specint, resolved_var_frac, waveno_cutoff_specint = spectrum_integration(
        psi,
        wavenumber,
        eps1,
        mol_visc,
        platform_speed,
        waveno_cutoff_spatial_corr,
        freq_cutoff_antialias,
        freq_cutoff_corrupt,
    )

    k_kolmogorov = (eps1 / mol_visc**3) ** 0.25
    eps_isrfit, waveno_cutoff_isrfit = inertial_range_fit(psi, wavenumber, k_kolmogorov)

    eps = np.where(eps1 < eps_crit, eps_specint, eps_isrfit)
    eps_source_flag = np.where(eps1 < eps_crit, 1, 2)

    waveno_cutoff = np.where(
        eps_source_flag == 1,
        waveno_cutoff_specint,
        waveno_cutoff_isrfit,
    )
    # do not count first element: psi(0)=0
    num_spec_points = (wavenumber[newaxis, :, :] <= waveno_cutoff[:, :, newaxis]).sum(
        axis=-1
    ) - 1
    kolm_length = (mol_visc[newaxis, :] ** 3 / eps) ** 0.25

    log_diss_var = get_log_diss_var(
        eps_source_flag,
        kolm_length,
        data_length,
        resolved_var_frac,
        num_spec_points,
        log_psi_var,
    )

    psi_model = model_spectrum(wavenumber, eps, mol_visc)
    use_wavenumber = wavenumber[newaxis, :, :] <= waveno_cutoff[:, :, newaxis]
    fom, log_diss_mad, num_spec_points_fom = figure_of_merit(
        np.where(use_wavenumber, psi, np.nan)[..., 1:],
        np.where(use_wavenumber, psi_model, np.nan)[..., 1:],
        log_psi_var,
    )
    assert np.all(np.equal(num_spec_points, num_spec_points_fom))  # Sanity check

    return (
        eps,
        eps_source_flag,
        log_diss_var,
        kolm_length,
        resolved_var_frac,
        fom,
        log_diss_mad,
        num_spec_points,
    )


def get_log_diss_var(
    eps_source_flag: Int[ndarray, "nshear time"],
    kolm_length: Float[ndarray, "nshear time"],
    data_length: Float[ndarray, "time"],
    resolved_var_frac: Float[ndarray, "nshear time"],
    num_spec_points: Int[ndarray, "nshear time"],
    log_psi_var: float,
) -> Float[ndarray, "nshear time"]:  # log(eps) variance
    """Notes:
    `eps_source_flag`: 1: spec_int, 2: isr_fit
    """
    data_length_nondim = data_length[newaxis, :] / kolm_length * resolved_var_frac**0.75
    log_diss_var_spec_int = 5.5 / (1 + (data_length_nondim / 4) ** (7 / 9))

    log_diss_var_isr_fit = 1.5 * log_psi_var / np.sqrt(num_spec_points)

    log_diss_var = np.where(
        eps_source_flag == 1,
        log_diss_var_spec_int,
        log_diss_var_isr_fit,
    )

    return log_diss_var


def figure_of_merit(
    psi: Float[ndarray, "nshear time wavenumber"],
    psi_model: Float[ndarray, "nshear time wavenumber"],
    log_psi_var: float,
) -> tuple[
    Float[ndarray, "nshear time"],
    Float[ndarray, "nshear time"],
    Int[ndarray, "nshear time"],
]:
    summand = np.abs(np.log(psi) - np.log(psi_model))
    num_spec_points = (~np.isnan(summand)).sum(axis=-1)
    log_diss_mad = np.nanmean(summand, axis=-1)
    tm = 0.8 + 1.25 / np.sqrt(num_spec_points)  # T_M in ATOMIX paper
    fom = log_diss_mad / log_psi_var / tm
    return fom, log_diss_mad, num_spec_points


def inertial_range_fit(
    psi: Float[ndarray, "nshear time wavenumber"],
    wavenumber: Float[ndarray, "time wavenumber"],
    k_kolmogorov: Float[ndarray, "nshear time"],
    a_kolmogorov: float = 8.19,
) -> tuple[
    Float[ndarray, "nshear time"],  # epsilon
    Float[ndarray, "nshear time"],  # wavenumber cutoff
]:
    """
    See Eq. 28. This assumes a known Kolmogorov constant, as opposed to linear
    regression to both slope and offset.

    The default value for A corresponds to Kolomgorov's constant for 1-dimensional
    spectra C1=0.53.
    """
    ln_epsilon = 1.5 * (
        np.log(psi) - np.log(a_kolmogorov) - np.log(wavenumber[newaxis, ...]) / 3
    )
    waveno_cutoff = 0.01 * k_kolmogorov
    ln_epsilon_fitrange = np.where(
        wavenumber[newaxis, ...] < waveno_cutoff[:, :, newaxis],
        ln_epsilon,
        np.nan,
    )
    eps = np.exp(np.nanmean(ln_epsilon_fitrange, axis=2))
    return eps, waveno_cutoff


def spectrum_integration(
    psi: Float[ndarray, "nshear time wavenumber"],
    wavenumber: Float[ndarray, "time wavenumber"],
    eps1: Float[ndarray, "nshear time"],
    mol_visc: Float[ndarray, "time"],
    platform_speed: Float[ndarray, "time"],
    waveno_cutoff_spatial_corr: float,
    freq_cutoff_antialias: float,
    freq_cutoff_corrupt: float,
) -> tuple[
    Float[ndarray, "nshear time"],
    Float[ndarray, "nshear time"],
    Float[ndarray, "nshear time"],
]:
    # 2nd estimate
    (eps2, waveno_cutoff) = get_eps_second_estimate(
        psi,
        wavenumber,
        eps1,
        mol_visc,
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
        # raise ValueError (waveno_cutoff.shape)
        eps /= get_spectral_variance_resolved_fraction(
            waveno_cutoff, kolmogorov_length(eps, mol_visc)
        )
        eps_increase = eps / eps_previous

    resolved_var_frac = get_spectral_variance_resolved_fraction(
        waveno_cutoff, kolmogorov_length(eps, mol_visc)
    )
    return eps, resolved_var_frac, waveno_cutoff


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
    waveno: Float[ndarray, "nshear time"],
    length_kolmogorov: Float[ndarray, "nshear time"],
) -> Float[ndarray, "nshear time"]:
    # Eq. 11; I_L (3rd model)
    k43 = (waveno * length_kolmogorov) ** (4.0 / 3.0)
    return np.tanh(65.5 * k43) - 9.0 * k43 * np.exp((-54.5 * k43))


def kolmogorov_length(
    eps: Float[ndarray, "*any"],
    mol_visc: Float[ndarray, "*any"],
) -> Float[ndarray, "*any"]:
    """The Kolmogorov length scale"""
    return (mol_visc**3 / eps) ** 0.25


def psi_nondim_factor(
    eps: Float[ndarray, "*any"],
    mol_visc: Float[ndarray, "*any"],
) -> Float[ndarray, "*any"]:
    """To pass from non-dimensional to dimensional shear spectra, see Eq. 6"""
    return (eps**3 / mol_visc) ** 0.25


def model_spectrum(
    k: Float[ndarray, "*any waveno"],
    eps: Float[ndarray, "*any"],
    mol_visc: Float[ndarray, "*any"],
):
    """Uses the Lueck spectrum (Eq. 9) - consistent with
    `get_spectral_variance_resolved_fraction`"""
    k_nondim = k[newaxis, :, :] / kolmogorov_length(eps, mol_visc)[:, :, newaxis]
    psi_nondim = model_spectrum_lueck(k_nondim)
    return psi_nondim * psi_nondim_factor(eps, mol_visc)[:, :, newaxis]


def model_spectrum_lueck(
    kn: Float[ndarray, "*any waveno"],  # nondimensional wavenumber
) -> Float[ndarray, "*any waveno"]:
    """Non-dimensional form, Eq. 9"""
    y = (kn / 0.015) ** 2
    fac1 = 8.048 * kn ** (1 / 3) / (1 + (21.7 * kn) ** 3)
    fac2 = 1 / (1 + (6.6 * kn) ** 2.5)
    fac3 = 1 + 0.36 * y / ((y - 1) ** 2 + 2 * y)
    return fac1 * fac2 * fac3
