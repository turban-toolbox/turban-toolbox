import numpy as np
from jaxtyping import Float, Int
from numpy import ndarray, newaxis
from scipy.signal import butter, freqz, lfilter, lfiltic
from scipy.special import erf, gamma

from turban.utils.util import integrate, reshape_any_first, reshape_halfoverlap_last

# nu, kin. viscosity of water; assumed known constant
viscosity_kinematic = 0.0000016
# molecular temperature diffusivity [m^2/s]
diffusivity_temp = 0.00000014
# constant for batchelor spectrum
q_b = 3.7


def microtemp(
    temp_emph: Float[ndarray, "time_fast"],
    senspeed: Float[ndarray, "time_fast"],
    section_select_idx: list[list[int]],
    sampfreq: float,
    segment_length: int,
    chunklen: int = 5,
    chunkoverlap: int = 2,
    outfile: str | None = None,
) -> tuple[Float[ndarray, "... time_slow"], ...]:
    """
    Process temperature microstructure.

    Default values to calculate spectra using 3+2 half-overlapping FFT
    intervals, i.e. 3*2048 samples.
    """

    dTdt = fft_grad(temp_emph, 1 / sampfreq)

    dTdt_segments = [dTdt[..., inds] for inds in section_select_idx]
    senspeed_segments = [senspeed[..., inds] for inds in section_select_idx]

    out = []

    for dTdt_segment, senspeed_segment in zip(dTdt_segments, senspeed_segments):
        (
            k,
            Pk,
            chi,
            k_batchelor_estimate,
        ) = temperature_dissipation(
            dTdt=dTdt_segment,
            senspeed=senspeed_segment,
            chunklen=chunklen,
            chunkoverlap=chunkoverlap,
            segment_length=segment_length,
            sampfreq=sampfreq,
        )

        out.append(
            xr.Dataset(
                data_vars={
                    "k": (["time_slow", "waveno"], k),
                    "Pk": (["time_slow", "waveno"], Pk),
                    # "Pnoise": (["time_slow", "waveno"], Pnoise),
                    "chi": (["time_slow"], chi),
                    "k_batchelor_estimate": (["time_slow"], k_batchelor_estimate),
                }
            )
        )

    ds = xr.concat(out, dim="time_slow")
    if outfile is not None:
        ds.to_netcdf(outfile, mode="a", group="microtemp")

    return (
        ds["chi"].values,
        ds["k_batchelor_estimate"].values,
    )


def tke_dissipation(
    k: Float[ndarray, "time waveno"],
    Pk: Float[ndarray, "time waveno"],
    k_batchelor_estimates: Float[ndarray, "time"],  # initial guess
):
    raise NotImplementedError


def temperature_dissipation(
    dTdt: Float[ndarray, "time_fast"],
    senspeed: Float[ndarray, "time_fast"],
    chunklen: int,
    chunkoverlap: int,
    segment_length: int,
    sampfreq: float,
    waveno_limit_upper: float,
) -> tuple[
    Float[ndarray, "time_slow waveno"],
    Float[ndarray, "time_slow waveno"],
    Float[ndarray, "time_slow waveno"],
    Float[ndarray, "time_slow"],
    Float[ndarray, "time_slow"],
]:
    """
    Calculate chi (temperature variance dissipation)
    """
    k, Pk, Pnoise = temperature_gradient_spectra(
        dTdt,
        senspeed,
        chunklen,
        chunkoverlap,
        segment_length,
        sampfreq,
    )

    chi = integrate_chi(k, Pk, Pnoise, waveno_limit_upper)

    # first round of k_batchelor estimates
    kb_est1 = k_batchelor_mle(chi, k, Pk, Pnoise)
    # correct for unresolved variance
    # note that k[:, 0]==0 is dropped from integrals
    kstar = 0.04 * np.sqrt(diffusivity_temp / viscosity_kinematic) * kb_est1
    # lower waveno integration limit
    kLO = k[range(k.shape[0]), np.argmax(3 * kstar[:, newaxis] <= k, axis=1)]
    kLO = np.where(
        3 * kstar >= k[:, 1], kLO, k[:, 1]
    )  # where k[1] is smaller than 3*kstar, round up to kstar
    # # higher waveno integration limit
    kUP = k[:, -1]
    chi_low = (
        6.0
        * diffusivity_temp
        * integrate_batchelor_theoretical(
            1e-3 + np.zeros_like(kb_est1),  # something "small"
            kLO,
            kb_est1,
            k,
            chi,
            diffusivity_temp,
            q_b,
        )
    )
    chi_high = (
        6.0
        * diffusivity_temp
        * integrate_batchelor_theoretical(
            kUP,
            2000.0 + np.zeros_like(kb_est1),
            kb_est1,
            k,
            chi,
            diffusivity_temp,
            q_b,
        )
    )
    chi = chi + chi_low + chi_high
    chi_correction = (chi_high + chi_low) / chi

    # MLE fitting, round 2
    kb_est2 = k_batchelor_mle(chi, k, Pk, Pnoise)

    # TODO missing steps
    # find maximum likelihood from 5th order polynomial fit -> kB (MOVE THIS INTO k_batchelor_mle...!!!)
    # find eps, eps_CI from that
    # calculate theoretical spectrum from chi, eps so far
    # integrate theoretical spectrum to give final value of chi
    # (sounds a bit hacky to me)
    # get linear fit to observed spectrum -> use for likelihood ratio

    return (
        k,
        Pk,
        Pnoise,
        chi,
        kb_est2,
    )


def k_batchelor_mle(
    chi: Float[ndarray, "time_slow"],
    k: Float[ndarray, "time_slow waveno"],
    Pk: Float[ndarray, "time_slow waveno"],
    Pnoise: Float[ndarray, "time_slow waveno"],
) -> Float[ndarray, "time_slow"]:
    # MLE fitting
    kmax = 3.0 * np.nanmax(k)
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
    cfunc_values: Float[ndarray, "kbatch time_slow"] = get_k_batchelor_costfunc(
        k_test,
        chi,
        k,
        Pk,
        Pnoise,
        viscosity_kinematic,
        diffusivity_temp,
    )
    k_batchelor_idx: Int[ndarray, "time_slow"] = np.argmax(cfunc_values, axis=0)
    k_batchelor_estimates = k_test[k_batchelor_idx]

    return k_batchelor_estimates


def integrate_chi(
    k: Float[ndarray, "time_slow waveno"],
    Pk: Float[ndarray, "time_slow waveno"],
    Pnoise: Float[ndarray, "time_slow waveno"],
    waveno_limit_upper: float = 500.0,
) -> Float[ndarray, "time_slow"]:
    # find integration limits
    is_above_noise = Pk > 2 * Pnoise
    waveno_above_noise = k[
        range(len(k)), -np.argmax(is_above_noise[:, ::-1], axis=1) - 1
    ]
    ku = np.minimum(waveno_above_noise, waveno_limit_upper)
    kl = k[:, 1]  # note that k[:, 0]==0 is dropped from integrals

    # first pass chi integration
    _spec_int: Float[ndarray, "time_slow"] = integrate(Pk, k, kl, ku)
    _spec_int = np.where(_spec_int >= 0, _spec_int, integrate(Pnoise, k, kl, ku))
    chi = 6 * diffusivity_temp * _spec_int
    return chi


def integrate_batchelor_theoretical(
    waveno_from: Float[ndarray, "time"],
    waveno_to: Float[ndarray, "time"],
    k_batchelor: Float[ndarray, "time"],
    k: Float[ndarray, "time frequency"],
    chi: Float[ndarray, "time"],
    diffusivity_temp: float,
    q_b: float,
) -> Float[ndarray, "time"]:
    psi: Float[ndarray, "time frequency"] = theoretical_spectrum(
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
    chi: Float[ndarray, "time"],
    waveno: Float[ndarray, "time waveno"],
    psi: Float[ndarray, "time waveno"],
    psi_noise: Float[ndarray, "time waveno"],
    viscosity_kinematic: float,
    diffusivity_temp: float,
) -> Float[ndarray, "kbatch time"]:
    psi_theoretical = (
        theoretical_spectrum(
            waveno,
            np.repeat(k_batchelor_test[:, newaxis], chi.shape[0], axis=1),
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
    k_batchelor: Float[ndarray, "kbatch time"],
    chi: Float[ndarray, "time"],
    viscosity_kinematic: float,
    diffusivity_temp: float,
    qB: float = 3.7,  # batchelor spectrum constant
) -> Float[ndarray, "kbatch time waveno"]:
    # TODO double check that k in radian units
    # 1D temperature gradient spectrum (Batchelor)
    # Eqn. 7 in Peterson et al. 2014

    eps = k_batchelor ** (4.0) * viscosity_kinematic * diffusivity_temp ** (2.0)

    # temp. gradient spectrum
    return (
        chi[newaxis, :, newaxis]
        * np.sqrt(viscosity_kinematic / eps[:, :, newaxis])
        / k_batchelor[:, :, newaxis]
        * (
            waveno[newaxis, ...] ** (2.0)
            * (
                qB
                * k_batchelor[:, :, newaxis]
                / waveno[newaxis, ...]
                * np.exp(
                    -qB
                    * waveno[newaxis, ...] ** (2.0)
                    / k_batchelor[:, :, newaxis] ** (2.0)
                )
                + np.sqrt(np.pi)
                * qB ** (1.5)
                * (
                    erf(np.sqrt(qB) * waveno[newaxis, ...] / k_batchelor[:, :, newaxis])
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


def get_noise(
    spectra: Float[ndarray, "time frequency"],
) -> Float[ndarray, "1 frequency"]:
    """
    Define noise as average of least intense 5% of spectra
    """
    if spectra.shape[0] > 0:
        # a measure of the intensity of the spectrum
        spec_intens = np.mean(spectra[:, :20], axis=1)
        # 5 % least intense spectra
        (ii,) = np.where(spec_intens < np.percentile(spec_intens, 5))
        noise = 10 ** np.mean(np.log10(spectra[ii, :]), axis=0)[newaxis, :]
        return noise
    else:
        return np.zeros((1, spectra.shape[1]), dtype=float)
