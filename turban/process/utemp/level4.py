import numpy as np
from jaxtyping import Float, Int, Bool
from numpy import ndarray, newaxis
from scipy.special import erfc
from scipy.stats import chi2 as chi2_dist

from turban.process.utemp.config import SpectrumModel
from turban.utils.util import integrate
from turban.utils.logging import get_logger

logger = get_logger(__name__)

# TODO LATER
# minimum number of spectral values above 2x noise (line 94 in MLE_Fit_Kb.m)
_MIN_POINTS_ABOVE_NOISE = 9
# number of trial k_B values in the fine search (lines 235-238 in MLE_Fit_Kb.m)
_N_TRIAL_FINE = 32
# half width (in trial points) of the polynomial fit (lines 293-299 in MLE_Fit_Kb.m)
_POLYFIT_HALFWIDTH = 7
# number of points on which the polynomial derivative is evaluated
_N_POLYFIT_GRID = 1001
_KBAT_SEARCH_POINTS: int = 100  # number of batchelor wave lengths


def temperature_dissipation(
    psi_k_gradt: Float[ndarray, "ntemp time waveno"],
    waveno: Float[ndarray, "time waveno"],
    psi_noise: Float[ndarray, "ntemp waveno"],
    waveno_limit_upper: float,
    molvisc: Float[ndarray, "time"],
    moldiff: Float[ndarray, "time"],
    spectrum_model: SpectrumModel,
    q: float,
    dof: float,
    n_trial: int = _KBAT_SEARCH_POINTS,
) -> tuple[
    Float[ndarray, "ntemp time"],  # chi
    Float[ndarray, "ntemp time"],  # eps
    Float[ndarray, "ntemp time"],  # eps_lower
    Float[ndarray, "ntemp time"],  # eps_upper
    Float[ndarray, "ntemp time"],  # kbatch
    Float[ndarray, "ntemp time"],  # kbatch_ci
    Float[ndarray, "ntemp time"],  # likelihood_ratio
    Float[ndarray, "ntemp time"],  # mad_ratio
    Float[ndarray, "ntemp time"],  # mad_ratio_pow
]:
    """Temperature variance dissipation (chi) and TKE dissipation (eps).

    Maximum likelihood fit of a theoretical spectrum to the temperature gradient
    spectrum, ported from ``MLE_Fit_Kb.m`` (Ruddick et al. 2000, Steinbuck et al.
    2009) via Ilker Fer.

    Parameters
    ----------
    psi_k_gradt : ndarray, shape (n_temp, time_slow, waveno)
        Temperature gradient power spectra in K2 m-2 cpm-1.
    waveno : ndarray, shape (time_slow, waveno)
        Wavenumber array in cpm.
    psi_noise : ndarray, shape (n_temp, waveno)
        Noise power spectrum in K2 m-2 cpm-1.
    waveno_limit_upper : float
        Hard upper wavenumber limit in cpm (``np.inf`` for none).
    molvisc : ndarray, shape (time_slow,)
        Kinematic molecular viscosity in m2 s-1.
    moldiff : ndarray, shape (time_slow,)
        Molecular thermal diffusivity in m2 s-1.
    spectrum_model : {"batchelor", "kraichnan"}
        Theoretical spectrum.
    q : float
        Universal constant of the theoretical spectrum.
    n_trial : int
        Number of trial k_B values in the rough search.
    dof : float
        Degrees of freedom of the spectral estimates.

    Returns
    -------
    chi : ndarray, shape (n_temp, time_slow)
        Temperature variance dissipation in K2 s-1.
    eps, eps_lower, eps_upper, kbatch, kbatch_ci, likelihood_ratio, mad_ratio, mad_ratio_pow
        See ``tke_dissipation``.

    All values are NaN where fewer than ``_MIN_POINTS_ABOVE_NOISE`` spectral values
    are above twice the noise.
    """
    waveno_upper = get_waveno_upper(waveno, psi_k_gradt, psi_noise, waveno_limit_upper)
    waveno_first = np.broadcast_to(waveno[:, 1], waveno_upper.shape)
    n_above_noise = _waveno_mask(waveno, waveno_first, waveno_upper).sum(axis=-1)
    is_valid = n_above_noise >= _MIN_POINTS_ABOVE_NOISE

    chi = integrate_chi(waveno, psi_k_gradt, psi_noise, waveno_upper, moldiff)

    # first pass coarse search for kbatch
    kb_est1, coarse_range = kbat_mle_coarse(
        chi,
        waveno,
        psi_k_gradt,
        psi_noise,
        waveno_first,
        waveno_upper,
        moldiff,
        spectrum_model,
        q,
        n_trial,
        dof,
    )

    # correct for unresolved variance
    # note that k[:, 0]==0 is dropped from integrals
    kstar: Float[ndarray, "n_temp time_slow"] = (
        0.04 * np.sqrt(moldiff / molvisc) * kb_est1
    )
    # lower waveno integration limit
    klo = waveno[
        range(waveno.shape[0]),
        np.argmax(3 * kstar[..., newaxis] <= waveno[newaxis, ...], axis=-1),
    ]
    klo: Float[ndarray, "n_temp time_slow"] = np.where(
        3 * kstar >= waveno[:, 1], klo, waveno[:, 1]
    )  # where k[1] is smaller than 3*kstar, round up to kstar
    chi_low = (
        6.0
        * moldiff
        * integrate_batchelor_theoretical(
            1e-3 + np.zeros_like(kb_est1),  # something "small"
            klo,
            kb_est1,
            waveno,
            chi,
            moldiff,
            spectrum_model,
            q,
        )
    )
    chi_high = (
        6.0
        * moldiff
        * integrate_batchelor_theoretical(
            waveno_upper,
            2000.0 + np.zeros_like(kb_est1),
            kb_est1,
            waveno,
            chi,
            moldiff,
            spectrum_model,
            q,
        )
    )
    chi = chi + chi_low + chi_high

    (
        kbatch,
        kbatch_ci,
        chi,
        likelihood_ratio,
        mad_ratio,
        mad_ratio_pow,
    ) = kbat_mle_fine(
        waveno,
        psi_k_gradt,
        kb_est1,
        coarse_range,
        chi,
        psi_noise,
        klo,
        waveno_upper,
        moldiff,
        spectrum_model,
        q,
        dof,
    )

    eps = eps_from_kbatch(kbatch, molvisc, moldiff)
    eps_lower = eps_from_kbatch(
        np.maximum(kbatch - kbatch_ci, 0.0), molvisc, moldiff
    )
    eps_upper = eps_from_kbatch(kbatch + kbatch_ci, molvisc, moldiff)


    out = (
        chi,
        eps,
        eps_lower,
        eps_upper,
        kbatch,
        kbatch_ci,
        likelihood_ratio,
        mad_ratio,
        mad_ratio_pow,
    )
    return tuple(np.where(is_valid, x, np.nan) for x in out)


def kbat_mle_fine(
    waveno: Float[ndarray, "time_slow waveno"],
    psi_k_gradt: Float[ndarray, "n_temp time_slow waveno"],
    kbatch_estimates: Float[ndarray, "n_temp time_slow"],
    search_halfwidth: Float[ndarray, "n_temp time_slow"],
    chi: Float[ndarray, "n_temp time_slow"],
    psi_noise: Float[ndarray, "n_temp waveno"],
    waveno_lower: Float[ndarray, "n_temp time_slow"],
    waveno_upper: Float[ndarray, "n_temp time_slow"],
    moldiff: Float[ndarray, "time_slow"],
    spectrum_model: SpectrumModel,
    q: float,
    dof: float,
) -> tuple[
    Float[ndarray, "n_temp time_slow"],  # kbatch
    Float[ndarray, "n_temp time_slow"],  # kbatch_ci
    Float[ndarray, "n_temp time_slow"],  # chi
    Float[ndarray, "n_temp time_slow"],  # likelihood_ratio
    Float[ndarray, "n_temp time_slow"],  # mad_ratio
    Float[ndarray, "n_temp time_slow"],  # mad_ratio_pow
]:
    """Maximum likelihood fit of the Batchelor wavenumber (fine search, 2nd pass).

    Parameters
    ----------
    waveno : ndarray, shape (time_slow, waveno)
        Wavenumber in cpm.
    psi_k_gradt : ndarray, shape (n_temp, time_slow, waveno)
        Observed temperature gradient spectra in K2 m-2 cpm-1.
    kbatch_estimates : ndarray, shape (n_temp, time_slow)
        Rough estimate of the Batchelor wavenumber in cpm (``kbatch_mle``).
    search_halfwidth : ndarray, shape (n_temp, time_slow)
        Half width of the fine search in cpm (``kbatch_mle``).
    chi : ndarray, shape (n_temp, time_slow)
        Temperature variance dissipation corrected for unresolved variance [K2 s-1].
    psi_noise : ndarray, shape (n_temp, waveno)
        Noise spectrum in K2 m-2 cpm-1.
    waveno_lower, waveno_upper : ndarray, shape (n_temp, time_slow)
        Fit range in cpm.
    molvisc : ndarray, shape (time_slow,)
        Kinematic molecular viscosity in m2 s-1.
    moldiff : ndarray, shape (time_slow,)
        Molecular thermal diffusivity in m2 s-1.
    spectrum_model : {"batchelor", "kraichnan"}
        Theoretical spectrum.
    q : float
        Universal constant of the theoretical spectrum.
    dof : float
        Degrees of freedom of the spectral estimates.

    Returns
    -------
    eps : ndarray, shape (n_temp, time_slow)
        TKE dissipation rate in W kg-1.
    eps_lower, eps_upper : ndarray, shape (n_temp, time_slow)
        eps at the lower/upper bound of the 95 % confidence interval of k_B.
    kbatch : ndarray, shape (n_temp, time_slow)
        Batchelor wavenumber in cpm.
    kbatch_ci : ndarray, shape (n_temp, time_slow)
        Half width of the 95 % confidence interval of k_B in cpm.
    chi : ndarray, shape (n_temp, time_slow)
        Temperature variance dissipation from the fitted spectrum in K2 s-1.
    likelihood_ratio : ndarray, shape (n_temp, time_slow)
        Likelihood of the fit relative to that of a power law.
    mad_ratio, mad_ratio_pow : ndarray, shape (n_temp, time_slow)
        Mean absolute deviation of psi / psi_fit for the fitted theoretical
        spectrum and for the power law.

    All values are NaN where the fine search fails.
    """
    fit_mask = _waveno_mask(waveno, waveno_lower, waveno_upper)
    psi_noise_ = psi_noise[:, newaxis, :]

    # 1. fine search
    k_lo = np.maximum(0.0, kbatch_estimates - search_halfwidth)
    k_hi = kbatch_estimates + search_halfwidth
    k_trial: Float[ndarray, "kbatch n_temp time_slow"] = np.linspace(
        k_lo, k_hi, _N_TRIAL_FINE
    )
    cost = get_kbatch_costfunc(
        k_trial,
        chi,
        waveno,
        psi_k_gradt,
        psi_noise,
        fit_mask,
        moldiff,
        spectrum_model,
        q,
        dof,
    )
    is_valid = np.any(np.isfinite(cost), axis=0)
    idx_max = np.argmax(cost, axis=0)
    likelihood = np.max(cost, axis=0)

    # polynomial refinement for numerical conditioning
    # TODO LATER fails for small k_B if the fine search window is wide compared to the
    # width of the likelihood peak (see kbatch_mle)
    dk = (k_hi - k_lo) / (_N_TRIAL_FINE - 1)
    k_mid = np.take_along_axis(k_trial, idx_max[newaxis], axis=0)[0]
    offset = np.arange(-_POLYFIT_HALFWIDTH, _POLYFIT_HALFWIDTH + 1)
    idx_fit = idx_max[..., newaxis] + offset
    in_range = (idx_fit >= 0) & (idx_fit < _N_TRIAL_FINE)
    y_fit = np.take_along_axis(
        np.moveaxis(cost, 0, -1), np.clip(idx_fit, 0, _N_TRIAL_FINE - 1), axis=-1
    )
    weight = (in_range & np.isfinite(y_fit)).astype(float)
    vander = offset.astype(float)[:, newaxis] ** np.arange(6)  # (offset, power)
    coef = (
        np.linalg.pinv(vander * weight[..., newaxis])
        @ np.where(weight > 0, y_fit, 0.0)[..., newaxis]
    )[..., 0]
    x_left = np.maximum(-_POLYFIT_HALFWIDTH, -idx_max)
    x_right = np.minimum(_POLYFIT_HALFWIDTH, _N_TRIAL_FINE - 1 - idx_max)
    x_grid = x_left[..., newaxis] + (x_right - x_left)[..., newaxis] * np.linspace(
        0, 1, _N_POLYFIT_GRID
    )
    dcoef = coef[..., 1:] * np.arange(1, 6)  # derivative
    ddcoef = dcoef[..., 1:] * np.arange(1, 5)  # second derivative
    x_best = np.take_along_axis(
        x_grid,
        np.argmin(np.abs(_polyval(dcoef, x_grid)), axis=-1)[..., newaxis],
        axis=-1,
    )[..., 0]
    kbatch = k_mid + x_best * dk
    with np.errstate(divide="ignore", invalid="ignore"):
        var_kbatch = -1 / (_polyval(ddcoef, x_best) / dk**2)
    kbatch_ci = 1.96 * np.sqrt(np.abs(var_kbatch))

    # power law fit and likelihood ratio
    with np.errstate(divide="ignore", invalid="ignore"):
        log_k = np.log(waveno)[newaxis]
        log_psi = np.log(psi_k_gradt)
    w = fit_mask & np.isfinite(log_k) & np.isfinite(log_psi)
    log_k, log_psi = np.where(w, log_k, 0.0), np.where(w, log_psi, 0.0)
    n, sx, sy = w.sum(axis=-1), log_k.sum(axis=-1), log_psi.sum(axis=-1)
    sxx, sxy = (log_k**2).sum(axis=-1), (log_k * log_psi).sum(axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        slope = (n * sxy - sx * sy) / (n * sxx - sx**2)
        intercept = (sy - slope * sx) / n
        psi_powerlaw = np.exp(intercept[..., newaxis]) * waveno ** slope[..., newaxis]
        likelihood_powerlaw = costfunction_c11(psi_k_gradt, psi_powerlaw, dof, fit_mask)
        likelihood_ratio = np.exp(likelihood - likelihood_powerlaw)

        # 6. chi from fitted spectrum (lines 339-364)
        psi_fit = theoretical_spectrum(
            spectrum_model,
            waveno[newaxis],
            kbatch[..., newaxis],
            chi[..., newaxis],
            moldiff[:, newaxis],
            q,
        )
        chi_fit = (
            6.0 * moldiff * integrate(psi_fit, waveno, waveno[:, 0], waveno[:, -1])
        )

        # 7. MAD (lines 367-370)
        mad_ratio = _mad_ratio(psi_k_gradt / (psi_fit + psi_noise_), fit_mask)
        mad_ratio_pow = _mad_ratio(psi_k_gradt / psi_powerlaw, fit_mask)

    out = (
        kbatch,
        kbatch_ci,
        chi_fit,
        likelihood_ratio,
        mad_ratio,
        mad_ratio_pow,
    )
    return tuple(np.where(is_valid, x, np.nan) for x in out)


def kbat_mle_coarse(
    chi: Float[ndarray, "n_temp time_slow"],
    waveno: Float[ndarray, "time_slow waveno"],
    psi_k: Float[ndarray, "n_temp time_slow waveno"],
    psi_noise: Float[ndarray, "n_temp waveno"],
    waveno_lower: Float[ndarray, "n_temp time_slow"],
    waveno_upper: Float[ndarray, "n_temp time_slow"],
    moldiff: Float[ndarray, "time_slow"],
    spectrum_model: SpectrumModel,
    q: float,
    n_trial: int,
    dof: float,
) -> tuple[
    Float[ndarray, "n_temp time_slow"],  # kbatch_estimates
    Float[ndarray, "n_temp time_slow"],  # search_halfwidth
]:
    """Rough maximum likelihood search for the Batchelor wavenumber.

    Evaluates the cost function C11 at ``n_trial`` trial values between
    ``3 max(waveno) / 600`` and ``3 max(waveno)`` and takes the maximum. The half
    width of the subsequent fine search is estimated from the curvature of C11 at
    its maximum (lines 217-227 in ``MLE_Fit_Kb.m``).

    Parameters
    ----------
    chi : ndarray, shape (n_temp, time_slow)
        Temperature variance dissipation in K2 s-1.
    waveno : ndarray, shape (time_slow, waveno)
        Wavenumber in cpm.
    psi_k : ndarray, shape (n_temp, time_slow, waveno)
        Observed temperature gradient spectra in K2 m-2 cpm-1.
    psi_noise : ndarray, shape (n_temp, waveno)
        Noise spectrum in K2 m-2 cpm-1.
    waveno_lower, waveno_upper : ndarray, shape (n_temp, time_slow)
        Fit range in cpm.
    moldiff : ndarray, shape (time_slow,)
        Molecular thermal diffusivity in m2 s-1.
    spectrum_model : {"batchelor", "kraichnan"}
        Theoretical spectrum.
    q : float
        Universal constant of the theoretical spectrum.
    n_trial : int
        Number of trial values.
    dof : float
        Degrees of freedom of the spectral estimates.

    Returns
    -------
    kbatch_estimates : ndarray, shape (n_temp, time_slow)
        Rough estimate of the Batchelor wavenumber in cpm.
    search_halfwidth : ndarray, shape (n_temp, time_slow)
        Half width of the fine search in cpm.
    """
    # TODO LATER this global grid is coarse for small k_B (step ~26 cpm at
    # max(waveno) ~ 850 cpm): with synthetic spectra, k_B = 30 cpm is not recovered by
    # the fine search/polynomial fit, while 150 and 400 cpm are. MLE_Fit_Kb.m uses a
    # a different grid over [max(k_fit) / 200, 3 max(k_fit)] for each chunk instead.
    # however, I am unsure how performance is affected by that. in principle we want to vectorize everything
    kmax = 3.0 * np.nanmax(waveno)
    kmin = kmax / 600.0
    k_test: Float[ndarray, "kbatch"] = np.linspace(kmin, kmax, n_trial)
    step = k_test[1] - k_test[0]

    cost: Float[ndarray, "kbatch n_temp time_slow"] = get_kbatch_costfunc(
        k_test[:, newaxis, newaxis],
        chi,
        waveno,
        psi_k,
        psi_noise,
        _waveno_mask(waveno, waveno_lower, waveno_upper),
        moldiff,
        spectrum_model,
        q,
        dof,
    )
    idx: Int[ndarray, "n_temp time_slow"] = np.argmax(cost, axis=0)
    kbatch_estimates = k_test[idx]

    # curvature of the cost function at its maximum
    def cost_at(i):
        return np.take_along_axis(cost, np.clip(i, 0, n_trial - 1)[newaxis], axis=0)[0]

    with np.errstate(divide="ignore", invalid="ignore"):
        delta_k = (2 * step) / np.sqrt(
            2 * cost_at(idx) - cost_at(idx - 1) - cost_at(idx + 1)
        )
    search_halfwidth = np.where(np.isfinite(delta_k), np.maximum(delta_k, step), step)
    return kbatch_estimates, search_halfwidth


def get_waveno_upper(
    waveno: Float[ndarray, "time_slow waveno"],
    psi_k: Float[ndarray, "n_temp time_slow waveno"],
    psi_noise: Float[ndarray, "n_temp waveno"],
    waveno_limit_upper: float,
) -> Float[ndarray, "n_temp time_slow"]:
    """Upper wavenumber limit: last wavenumber where the spectrum exceeds twice the
    noise, at most ``waveno_limit_upper``. ``waveno[:, 0]`` if nowhere above noise.

    Parameters
    ----------
    waveno : ndarray, shape (time_slow, waveno)
        Wavenumber in cpm.
    psi_k : ndarray, shape (n_temp, time_slow, waveno)
        Temperature gradient spectra.
    psi_noise : ndarray, shape (n_temp, waveno)
        Noise spectrum.
    waveno_limit_upper : float
        Hard upper limit in cpm.

    Returns
    -------
    ndarray, shape (n_temp, time_slow)
        Upper wavenumber limit in cpm.
    """
    is_above_noise = psi_k > 2 * psi_noise[..., newaxis, :]
    waveno_above_noise = waveno[
        range(waveno.shape[0]), -np.argmax(is_above_noise[..., ::-1], axis=-1) - 1
    ]
    waveno_above_noise = np.where(
        is_above_noise.any(axis=-1), waveno_above_noise, waveno[:, 0]
    )
    return np.minimum(waveno_above_noise, waveno_limit_upper)


def integrate_chi(
    waveno: Float[ndarray, "time_slow waveno"],
    psi_k: Float[ndarray, "n_temp time_slow waveno"],
    psi_noise: Float[ndarray, "n_temp waveno"],
    waveno_upper: Float[ndarray, "n_temp time_slow"],
    moldiff: Float[ndarray, "time_slow"],
) -> Float[ndarray, "n_temp time_slow"]:
    """First pass chi = 6 D int psi dk, from ``waveno[:, 1]`` to ``waveno_upper``.

    Parameters
    ----------
    waveno : ndarray, shape (time_slow, waveno)
        Wavenumber in cpm.
    psi_k : ndarray, shape (n_temp, time_slow, waveno)
        Temperature gradient spectra in K2 m-2 cpm-1.
    psi_noise : ndarray, shape (n_temp, waveno)
        Noise spectrum, integrated instead where the integral of ``psi_k`` is
        negative.
    waveno_upper : ndarray, shape (n_temp, time_slow)
        Upper integration limit in cpm (``get_waveno_upper``).
    moldiff : ndarray, shape (time_slow,)
        Molecular thermal diffusivity in m2 s-1.

    Returns
    -------
    ndarray, shape (n_temp, time_slow)
        Temperature variance dissipation in K2 s-1.
    """
    kl = np.broadcast_to(
        waveno[:, 1], waveno_upper.shape
    )  # note that k[:, 0]==0 is dropped from integrals
    _spec_int: Float[ndarray, "n_temp time_slow"] = integrate(
        psi_k, waveno, kl, waveno_upper
    )
    _spec_int = np.where(
        _spec_int >= 0,
        _spec_int,
        integrate(psi_noise[:, newaxis, :], waveno, kl, waveno_upper),
    )
    return 6 * moldiff * _spec_int


def integrate_batchelor_theoretical(
    waveno_from: Float[ndarray, "n_temp time"],
    waveno_to: Float[ndarray, "n_temp time"],
    kbatch: Float[ndarray, "n_temp time"],
    waveno: Float[ndarray, "time waveno"],
    chi: Float[ndarray, "n_temp time"],
    moldiff: Float[ndarray, "time"],
    spectrum_model: SpectrumModel,
    q: float,
) -> Float[ndarray, "n_temp time"]:
    """Integral of the theoretical spectrum over ``waveno`` (cpm) within limits."""
    psi: Float[ndarray, "n_temp time waveno"] = theoretical_spectrum(
        spectrum_model,
        waveno[newaxis],
        kbatch[..., newaxis],
        chi[..., newaxis],
        moldiff[:, newaxis],
        q,
    )
    return integrate(psi, waveno, waveno_from, waveno_to)


def get_kbatch_costfunc(
    kbatch_test: Float[ndarray, "kbatch #n_temp #time"],
    chi: Float[ndarray, "n_temp time"],
    waveno: Float[ndarray, "time waveno"],
    psi: Float[ndarray, "n_temp time waveno"],
    psi_noise: Float[ndarray, "n_temp waveno"],
    fit_mask: Bool[ndarray, "n_temp time waveno"],
    moldiff: Float[ndarray, "time"],
    spectrum_model: SpectrumModel,
    q: float,
    dof: float,
) -> Float[ndarray, "kbatch n_temp time"]:
    """Cost function C11 for trial Batchelor wavenumbers, -inf where undefined.

    The model spectrum is the theoretical spectrum plus noise.
    """
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        psi_theoretical = (
            theoretical_spectrum(
                spectrum_model,
                waveno[newaxis, newaxis],
                kbatch_test[..., newaxis],
                chi[newaxis, ..., newaxis],
                moldiff[:, newaxis],
                q,
            )
            + psi_noise[:, newaxis, :]
        )
        values = costfunction_c11(psi, psi_theoretical, dof, fit_mask)
    return np.where(np.isfinite(values), values, -np.inf)


def costfunction_c11(
    psi: Float[ndarray, "*#batch time waveno"],  # observed spectrum
    psi_theoretical: Float[ndarray, "*#batch time waveno"],  # model spectrum
    dof: float,
    fit_mask: Bool[ndarray, "*#batch time waveno"],
) -> Float[ndarray, "*batch time"]:
    """Cost function C11: mean log-likelihood of the observed spectrum.

    C11 = mean(ln(chi2_pdf(dof psi / psi_t, dof) dof / psi_t + eps)) over the
    spectral values in ``fit_mask``, with eps the machine epsilon. Eq. 18 in
    Ruddick et al. (2000), divided by the number of spectral values as in
    Steinbuck et al. (2009).

    Parameters
    ----------
    psi : ndarray
        Observed spectrum.
    psi_theoretical : ndarray
        Model spectrum, broadcast against ``psi``.
    dof : float
        Degrees of freedom of the spectral estimates.
    fit_mask : ndarray of bool
        Spectral values to use.

    Returns
    -------
    ndarray
        Cost function value, NaN where ``fit_mask`` is all False.
    """
    log_l = np.log(
        chi2_dist.pdf(dof * psi / psi_theoretical, dof) * dof / psi_theoretical
        + np.finfo(float).eps
    )
    return np.sum(np.where(fit_mask, log_l, 0.0), axis=-1) / np.sum(fit_mask, axis=-1)


def eps_from_kbatch(
    kbatch: Float[ndarray, "..."],
    molvisc: Float[ndarray, "..."] | float,
    moldiff: Float[ndarray, "..."] | float,
) -> Float[ndarray, "..."]:
    """TKE dissipation eps = nu D^2 (2 pi k_B)^4, k_B in cpm (Batchelor 1959)."""
    return molvisc * moldiff**2 * (2 * np.pi * kbatch) ** 4


def _waveno_mask(
    waveno: Float[ndarray, "time waveno"],
    waveno_lower: Float[ndarray, "n_temp time"],
    waveno_upper: Float[ndarray, "n_temp time"],
) -> Bool[ndarray, "n_temp time waveno"]:
    """True where ``waveno_lower <= waveno <= waveno_upper``."""
    return (waveno_lower[..., newaxis] <= waveno) & (
        waveno <= waveno_upper[..., newaxis]
    )


def _polyval(coef: Float[ndarray, "*batch npow"], x: Float[ndarray, "..."]):
    """Polynomial with increasing powers ``coef``, broadcast over leading dims."""
    if x.ndim > coef.ndim - 1:  # x has an extra trailing dim
        coef = coef[..., newaxis, :]
    return np.sum(coef * x[..., newaxis] ** np.arange(coef.shape[-1]), axis=-1)


def _mad_ratio(ratio: Float[ndarray, "n_temp time waveno"], fit_mask) -> ndarray:
    """Mean absolute deviation of ``ratio`` from its mean, within ``fit_mask``."""
    n = fit_mask.sum(axis=-1)
    mean = np.sum(np.where(fit_mask, ratio, 0.0), axis=-1) / n
    dev = np.abs(ratio - mean[..., newaxis])
    return np.sum(np.where(fit_mask, dev, 0.0), axis=-1) / n


def _shape_batchelor(y):
    """Dimensionless shape of the Batchelor (1959) gradient spectrum.

    ``y = k sqrt(2q) / k_B`` and the shape are dimensionless; the dimensional
    prefactor is applied in ``theoretical_spectrum`` (Sanchez et al. 2011).
    """
    return y * (np.exp(-(y**2) / 2) - y * np.sqrt(np.pi / 2) * erfc(y / np.sqrt(2)))


def _shape_kraichnan(y):
    """Dimensionless shape of the Kraichnan (1968) gradient spectrum.

    ``y = k sqrt(2q) / k_B`` and the shape are dimensionless; the dimensional
    prefactor is applied in ``theoretical_spectrum`` (Sanchez et al. 2011).
    """
    return y * np.exp(-np.sqrt(3) * y)


_SHAPES = {"batchelor": _shape_batchelor, "kraichnan": _shape_kraichnan}


def theoretical_spectrum(
    model: SpectrumModel,
    waveno: Float[ndarray, "..."],
    kbatch: Float[ndarray, "..."] | float,
    chi: Float[ndarray, "..."] | float,
    moldiff: Float[ndarray, "..."] | float,
    q: float,
) -> Float[ndarray, "..."]:
    """One-dimensional temperature gradient spectrum [K2 m-2 cpm-1].

    Batchelor (1959) or Kraichnan (1968) form, written as in Sanchez et al. (2011)
    and Piccolroaz et al. (2021, eq. 13-14). Both integrate to chi / (6 D).

    Parameters
    ----------
    model : {"batchelor", "kraichnan"}
        Spectral form.
    waveno : ndarray
        Wavenumber [cpm].
    kbatch : ndarray or float
        Batchelor wavenumber [cpm], broadcast against `waveno`.
    chi : ndarray or float
        Dissipation rate of temperature variance [K2 s-1].
    moldiff : ndarray or float
        Molecular thermal diffusivity [m2 s-1].
    q : float
        Universal constant of the spectrum.
    """
    y = waveno * np.sqrt(2 * q) / kbatch
    return chi / (2 * moldiff * kbatch) * np.sqrt(2 * q) * _SHAPES[model](y)
