import numpy as np
from jaxtyping import Float, Int
from numpy import ndarray, newaxis
from scipy.signal import butter, freqz, lfilter, lfiltic
from scipy.special import erf, gamma

from turban.utils.util import integrate, reshape_any_first, reshape_halfoverlap_last
from turban.utils.logging import logger

# nu, kin. viscosity of water; assumed known constant
viscosity_kinematic = 0.0000016
# molecular temperature diffusivity [m^2/s]
diffusivity_temp = 0.00000014
# constant for batchelor spectrum
q_b = 3.7


def tke_dissipation(
    k: Float[ndarray, "time waveno"],
    Pk: Float[ndarray, "time waveno"],
    k_batchelor_estimates: Float[ndarray, "time"],  # initial guess
):
    raise NotImplementedError


def temperature_dissipation(
    psi_k: Float[ndarray, "n_temp time_slow waveno"],
    waveno: Float[ndarray, "time_slow waveno"],
    psi_noise: Float[ndarray, "n_temp waveno"],
    waveno_limit_upper: float,
) -> tuple[
    Float[ndarray, "n_temp time_slow"],
    Float[ndarray, "n_temp time_slow"],
]:
    """
    Calculate chi (temperature variance dissipation)
    """
    logger.critical('Missing some parts of algorithm. Use with care!')

    chi = integrate_chi(waveno, psi_k, psi_noise, waveno_limit_upper)

    # first round of k_batchelor estimates
    kb_est1: Float[ndarray, "n_temp time_slow"]
    kb_est1 = k_batchelor_mle(chi, waveno, psi_k, psi_noise)
    # correct for unresolved variance
    # note that k[:, 0]==0 is dropped from integrals
    kstar: Float[ndarray, "n_temp time_slow"] = (
        0.04 * np.sqrt(diffusivity_temp / viscosity_kinematic) * kb_est1
    )
    # lower waveno integration limit
    klo = waveno[
        range(waveno.shape[0]),
        np.argmax(3 * kstar[..., newaxis] <= waveno[newaxis, ...], axis=-1),
    ]
    klo: Float[ndarray, "n_temp time_slow"] = np.where(
        3 * kstar >= waveno[:, 1], klo, waveno[:, 1]
    )  # where k[1] is smaller than 3*kstar, round up to kstar
    # upper waveno integration limit
    kup: Float[ndarray, "n_temp time_slow"] = waveno[newaxis, :, -1]
    chi_low = (
        6.0
        * diffusivity_temp
        * integrate_batchelor_theoretical(
            1e-3 + np.zeros_like(kb_est1),  # something "small"
            klo,
            kb_est1,
            waveno,
            chi,
            diffusivity_temp,
            q_b,
        )
    )
    chi_high = (
        6.0
        * diffusivity_temp
        * integrate_batchelor_theoretical(
            kup,
            2000.0 + np.zeros_like(kb_est1),
            kb_est1,
            waveno,
            chi,
            diffusivity_temp,
            q_b,
        )
    )
    chi = chi + chi_low + chi_high
    chi_correction = (chi_high + chi_low) / chi

    # MLE fitting, round 2
    kb_est2 = k_batchelor_mle(chi, waveno, psi_k, psi_noise)
    eps_est2 = kb_est2 ** (4.0) * viscosity_kinematic * diffusivity_temp ** (2.0)

    # TODO missing steps
    # find maximum likelihood from 5th order polynomial fit -> kB (move this into k_batchelor_mle?)
    # find eps, eps_CI from that
    # calculate theoretical spectrum from chi, eps so far
    # integrate theoretical spectrum to give final value of chi
    # (sounds a bit hacky to me)
    # get linear fit to observed spectrum -> use for likelihood ratio

    return (
        chi,
        eps_est2,
    )


def k_batchelor_mle(
    chi: Float[ndarray, "n_temp time_slow"],
    waveno: Float[ndarray, "time_slow waveno"],
    psi_k: Float[ndarray, "n_temp time_slow waveno"],
    psi_noise: Float[ndarray, "n_temp waveno"],
) -> Float[ndarray, "n_temp time_slow"]:
    # MLE fitting
    kmax = 3.0 * np.nanmax(waveno)
    kmin = kmax / 600.0
    k_test: Float[ndarray, "kbatch"] = np.linspace(
        kmin, kmax, 100
    )  # k_batchelor values to test

    # assert chi.shape[0] == k.shape[0] == Pk.shape[0]
    # assert k.shape[1] == Pk.shape[1]
    k_batchelor_idx = np.zeros_like(chi, dtype=int)
    # TODO: do not hardcode number of spectral points
    # cfunc_values_1: Float[ndarray, "time_slow k_test"] = np.zeros(
    #     (chi.shape[0], 30), dtype=float
    # )
    # TODO: speed up
    # for i in range(chi.shape[0]):
    cfunc_values: Float[ndarray, "kbatch n_temp time_slow"] = get_k_batchelor_costfunc(
        k_test,
        chi,
        waveno,
        psi_k,
        psi_noise,
        viscosity_kinematic,
        diffusivity_temp,
    )
    k_batchelor_idx: Int[ndarray, "n_temp time_slow"] = np.argmax(cfunc_values, axis=0)
    k_batchelor_estimates = k_test[k_batchelor_idx]

    return k_batchelor_estimates


def integrate_chi(
    waveno: Float[ndarray, "time_slow waveno"],
    psi_k: Float[ndarray, "n_temp time_slow waveno"],
    Pnoise: Float[ndarray, "n_temp waveno"],
    waveno_limit_upper: float,
) -> Float[ndarray, "n_temp time_slow"]:
    # find integration limits
    is_above_noise = psi_k > 2 * Pnoise[..., newaxis, :]
    waveno_above_noise = waveno[
        range(waveno.shape[0]), -np.argmax(is_above_noise[:, ::-1], axis=-1) - 1
    ]
    ku = np.minimum(waveno_above_noise, waveno_limit_upper)
    kl = waveno[:, 1]  # note that k[:, 0]==0 is dropped from integrals

    # first pass chi integration
    _spec_int: Float[ndarray, "n_temp time_slow"] = integrate(psi_k, waveno, kl, ku)
    _spec_int = np.where(_spec_int >= 0, _spec_int, integrate(Pnoise, waveno, kl, ku))
    chi = 6 * diffusivity_temp * _spec_int
    return chi


def integrate_batchelor_theoretical(
    waveno_from: Float[ndarray, "n_temp time"],
    waveno_to: Float[ndarray, "n_temp time"],
    k_batchelor: Float[ndarray, "n_temp time"],
    k: Float[ndarray, " time frequency"],
    chi: Float[ndarray, "n_temp time"],
    diffusivity_temp: float,
    q_b: float,
) -> Float[ndarray, "n_temp time"]:
    psi: Float[ndarray, "n_temp time frequency"] = theoretical_spectrum(
        k,
        k_batchelor[newaxis, :],  # create 1st axis of length 1
        chi,
        viscosity_kinematic,
        diffusivity_temp,
        q_b,
    ).squeeze(
        axis=0
    )  # get rid of kbatch axis (length 1)
    return integrate(psi, k, waveno_from, waveno_to)


def get_k_batchelor_costfunc(
    k_batchelor_test: Float[ndarray, "kbatch"],
    chi: Float[ndarray, "n_temp time"],
    waveno: Float[ndarray, "time waveno"],
    psi: Float[ndarray, "n_temp time waveno"],
    psi_noise: Float[ndarray, "n_temp waveno"],
    viscosity_kinematic: float,
    diffusivity_temp: float,
) -> Float[ndarray, "kbatch n_temp time"]:
    psi_theoretical = (
        theoretical_spectrum(
            waveno,
            np.repeat(k_batchelor_test[:, newaxis, newaxis], chi.shape[-1], axis=-1),
            chi,
            viscosity_kinematic,
            diffusivity_temp,
        )
        + psi_noise
    )
    values = costfunction_c11(psi, psi_theoretical)

    return values


def theoretical_spectrum(
    waveno: Float[ndarray, "time waveno"],
    k_batchelor: Float[ndarray, "kbatch n_temp time"],
    chi: Float[ndarray, "n_temp time"],
    viscosity_kinematic: float,
    diffusivity_temp: float,
    qB: float = 3.7,  # batchelor spectrum constant
) -> Float[ndarray, "kbatch n_temp time waveno"]:
    # TODO double check that k in radian units
    # 1D temperature gradient spectrum (Batchelor)
    # Eqn. 7 in Peterson et al. 2014

    eps: Float[ndarray, "kbatch n_temp time"]
    eps = k_batchelor ** (4.0) * viscosity_kinematic * diffusivity_temp ** (2.0)

    # temp. gradient spectrum
    return (
        chi[newaxis, :, :, newaxis]
        * np.sqrt(viscosity_kinematic / eps[..., newaxis])
        / k_batchelor[..., newaxis]
        * (
            waveno[newaxis, ...] ** (2.0)
            * (
                qB
                * k_batchelor[..., newaxis]
                / waveno[newaxis, ...]
                * np.exp(
                    -qB
                    * waveno[newaxis, ...] ** (2.0)
                    / k_batchelor[..., newaxis] ** (2.0)
                )
                + np.sqrt(np.pi)
                * qB ** (1.5)
                * (
                    erf(np.sqrt(qB) * waveno[newaxis, ...] / k_batchelor[..., newaxis])
                    - 1.0
                )
            )
        )
    )


def chisquared(x: Float[ndarray, "..."], dof: int):
    """
    pdf of chi^2 distribution
    """
    return x ** (dof / 2) * np.exp(-x / 2) / (2 ** (dof / 2) * gamma(dof / 2))


def costfunction_c11(
    psi: Float[ndarray, "... time waveno"],  # observed spectrum
    psi_theoretical: Float[ndarray, "... time waveno"],  # theoretical spectrum
) -> Float[ndarray, "... time"]:
    dof = 6
    # degrees of freedom
    cost_vector = chisquared(dof * psi / psi_theoretical, dof)
    return np.nanmean(cost_vector, axis=-1)
