import numpy as np
from jaxtyping import Float, Int
from numpy import ndarray, newaxis
from scipy.signal import butter, freqz, lfilter, lfiltic
from scipy.special import erf, gamma

from turban.util import integrate, reshape_any_first, reshape_halfoverlap_last

# nu, kin. viscosity of water; assumed known constant
viscosity_kinematic = 0.0000016
# molecular temperature diffusivity [m^2/s]
diffusivity_temp = 0.00000014
# constant for batchelor spectrum
q_b = 3.7


def microtemp(
    temp_emph: Float[ndarray, "time_fast"],
    pspd: Float[ndarray, "time_fast"],
    section_select_idx: list[list[int]],
    sampling_freq: float,
    fft_length: int,
    chunklen: int = 5,
    chunkoverlap: int = 2,
    outfile: str | None = None,
) -> tuple[Float[ndarray, "... time_slow"], ...]:
    """
    Process temperature microstructure.

    Default values to calculate spectra using 3+2 half-overlapping FFT
    intervals, i.e. 3*2048 samples.
    """

    dTdt = fft_grad(temp_emph, 1 / sampling_freq)

    dTdt_segments = [dTdt[..., inds] for inds in section_select_idx]
    pspd_segments = [pspd[..., inds] for inds in section_select_idx]

    out = []

    for dTdt_segment, pspd_segment in zip(dTdt_segments, pspd_segments):
        (
            k,
            Pk,
            chi,
            k_batchelor_estimate,
        ) = temperature_dissipation(
            dTdt=dTdt_segment,
            pspd=pspd_segment,
            chunklen=chunklen,
            chunkoverlap=chunkoverlap,
            fft_length=fft_length,
            sampling_freq=sampling_freq,
        )

        out.append(
            xr.Dataset(
                data_vars={
                    "k": (["time_slow", "wavenumber"], k),
                    "Pk": (["time_slow", "wavenumber"], Pk),
                    # "Pnoise": (["time_slow", "wavenumber"], Pnoise),
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
    k: Float[ndarray, "time wavenumber"],
    Pk: Float[ndarray, "time wavenumber"],
    k_batchelor_estimates: Float[ndarray, "time"],  # initial guess
):
    raise NotImplementedError


def temperature_dissipation(
    dTdt: Float[ndarray, "time_fast"],
    pspd: Float[ndarray, "time_fast"],
    chunklen: int,
    chunkoverlap: int,
    fft_length: int,
    sampling_freq: float,
    wavenumber_limit_upper: float = 500.0,
) -> tuple[
    Float[ndarray, "time_slow wavenumber"],
    Float[ndarray, "time_slow wavenumber"],
    Float[ndarray, "time_slow"],
    Float[ndarray, "time_slow"],
]:
    """
    Calculate chi (temperature variance dissipation)
    """
    k, Pk, Pnoise = temperature_gradient_spectra(
        dTdt,
        pspd,
        chunklen,
        chunkoverlap,
        fft_length,
        sampling_freq,
    )

    chi = integrate_chi(k, Pk, Pnoise, wavenumber_limit_upper)

    k_batchelor_estimates_1 = k_batchelor_mle(chi, k, Pk, Pnoise)
    # correct for unresolved variance
    # note that k[:, 0]==0 is dropped from integrals
    kstar = (
        0.04 * np.sqrt(diffusivity_temp / viscosity_kinematic) * k_batchelor_estimates_1
    )
    # lower wavenumber integration limit
    kLO = k[range(k.shape[0]), np.argmax(3 * kstar[:, newaxis] <= k, axis=1)]
    kLO = np.where(
        3 * kstar >= k[:, 1], kLO, k[:, 1]
    )  # where k[1] is smaller than 3*kstar, round up to kstar
    # # higher wavenumber integration limit
    kUP = k[:, -1]
    chi_low = (
        6.0
        * diffusivity_temp
        * integrate_batchelor_theoretical(
            1e-3 + np.zeros_like(k_batchelor_estimates_1),  # something "small"
            kLO,
            k_batchelor_estimates_1,
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
            2000.0 + np.zeros_like(k_batchelor_estimates_1),
            k_batchelor_estimates_1,
            k,
            chi,
            diffusivity_temp,
            q_b,
        )
    )
    chi = chi + chi_low + chi_high
    chi_correction = (chi_high + chi_low) / chi

    # MLE fitting, round 2
    k_batchelor_estimates_2 = k_batchelor_mle(chi, k, Pk, Pnoise)

    # TODO missing steps:
    # find maximum likelihood from 5th order polynomial fit -> kB (MOVE THIS INTO k_batchelor_mle...!!!)
    # find eps, eps_CI from that
    # calculate theoretical spectrum from chi, eps so far
    # integrate theoretical spectrum to give final value of chi
    # (sounds a bit hacky to me)
    # get linear fit to observed spectrum -> use for likelihood ratio

    return (
        k,
        Pk,
        chi,
        k_batchelor_estimates_2,
    )


def k_batchelor_mle(
    chi: Float[ndarray, "time_slow"],
    k: Float[ndarray, "time_slow wavenumber"],
    Pk: Float[ndarray, "time_slow wavenumber"],
    Pnoise: Float[ndarray, "time_slow wavenumber"],
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
    # raise NotImplementedError('TODO do search for MLE max here')
    return k_batchelor_estimates


def temperature_gradient_spectra(
    dTdt: Float[ndarray, "time_fast"],
    pspd: Float[ndarray, "time_fast"],
    chunklen: int,
    chunkoverlap: int,
    fft_length: int,
    sampling_freq: float,
) -> tuple[
    Float[ndarray, "time_slow wavenumber"],
    Float[ndarray, "time_slow wavenumber"],
    Float[ndarray, "1 wavenumber"],
]:
    yr: Float[ndarray, "time_slow freq"] = reshape_halfoverlap_last(dTdt, fft_length)
    yr -= yr.mean(axis=1)[:, np.newaxis]
    yr *= np.hanning(fft_length)[np.newaxis, :]

    freq: Float[ndarray, "1 freq"] = np.fft.rfftfreq(fft_length, d=1 / sampling_freq)[
        np.newaxis, :
    ]
    Fyr = np.fft.rfft(yr)[:, :]
    Pf: Float[ndarray, "time_slow freq"] = (Fyr.conj() * Fyr).real

    pspda: Float[ndarray, "time_slow 1"] = reshape_any_first(
        reshape_halfoverlap_last(pspd, fft_length).mean(axis=1)[:, np.newaxis],
        chunklen,
        chunkoverlap,
    ).mean(axis=1)
    assert pspda.shape[1] == 1

    correction = correction_frequency_response_bilinear(
        freq=freq, Fs=sampling_freq
    ) * correction_frequency_response_vachon_lueck(freq=freq, pspd=pspda)

    # average spectra by chunks
    Pf = reshape_any_first(Pf, chunklen, chunkoverlap).mean(axis=1)
    # Pf = Pf * correction

    k = freq / pspda
    Pk: Float[ndarray, "time_slow freq"] = Pf * pspda / fft_length / (sampling_freq / 2)

    Pnoise = get_noise(Pk)
    # Pk -= Pnoise

    return k, Pk, Pnoise


def integrate_chi(
    k: Float[ndarray, "time_slow wavenumber"],
    Pk: Float[ndarray, "time_slow wavenumber"],
    Pnoise: Float[ndarray, "time_slow wavenumber"],
    wavenumber_limit_upper: float = 500.0,
) -> Float[ndarray, "time_slow"]:
    # find integration limits
    is_above_noise = Pk > 2 * Pnoise
    wavenumber_above_noise = k[
        range(len(k)), -np.argmax(is_above_noise[:, ::-1], axis=1) - 1
    ]
    ku = np.minimum(wavenumber_above_noise, wavenumber_limit_upper)
    kl = k[:, 1]  # note that k[:, 0]==0 is dropped from integrals

    # first pass chi integration
    _spec_int: Float[ndarray, "time_slow"] = integrate(Pk, k, kl, ku)
    _spec_int = np.where(_spec_int >= 0, _spec_int, integrate(Pnoise, k, kl, ku))
    chi = 6 * diffusivity_temp * _spec_int
    return chi


def integrate_batchelor_theoretical(
    wavenumber_from: Float[ndarray, "time"],
    wavenumber_to: Float[ndarray, "time"],
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
    return integrate(psi, k, wavenumber_from, wavenumber_to)


def get_k_batchelor_costfunc(
    k_batchelor_test: Float[ndarray, "kbatch"],
    chi: Float[ndarray, "time"],
    wavenumber: Float[ndarray, "time wavenumber"],
    psi: Float[ndarray, "time wavenumber"],
    psi_noise: Float[ndarray, "time wavenumber"],
    viscosity_kinematic: float,
    diffusivity_temp: float,
) -> Float[ndarray, "kbatch time"]:
    psi_theoretical = (
        theoretical_spectrum(
            wavenumber,
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
    wavenumber: Float[ndarray, "time wavenumber"],
    k_batchelor: Float[ndarray, "kbatch time"],
    chi: Float[ndarray, "time"],
    viscosity_kinematic: float,
    diffusivity_temp: float,
    qB: float = 3.7,  # batchelor spectrum constant
) -> Float[ndarray, "kbatch time wavenumber"]:
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
            wavenumber[newaxis, ...] ** (2.0)
            * (
                qB
                * k_batchelor[:, :, newaxis]
                / wavenumber[newaxis, ...]
                * np.exp(
                    -qB
                    * wavenumber[newaxis, ...] ** (2.0)
                    / k_batchelor[:, :, newaxis] ** (2.0)
                )
                + np.sqrt(np.pi)
                * qB ** (1.5)
                * (
                    erf(
                        np.sqrt(qB)
                        * wavenumber[newaxis, ...]
                        / k_batchelor[:, :, newaxis]
                    )
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
    psi: Float[ndarray, "... time wavenumber"],  # observed spectrum
    psi_theoretical: Float[ndarray, "... time wavenumber"],  # theoretical spectrum
) -> Float[ndarray, "... time"]:
    dof = 6
    # degrees of freedom
    cost_vector = chisquared(dof * psi / psi_theoretical, dof)
    return np.nanmean(cost_vector, axis=-1)


def flatten01(x: ndarray):
    """
    Flatten first two dimensions of array
    """
    return x.reshape((x.shape[0] * x.shape[1], x.shape[2]))


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


def deconvolute_mss_ntchp(
    x: Int[ndarray, "time"],
    x_emph: Int[ndarray, "time"],
    sampling_freq: float,
    gain: float = 1.5,
):
    cutoff_freq_Hz = 1 / (2 * np.pi * gain)
    # cutoff_freq_Hz = 0.5
    cutoff_nondim = cutoff_freq_Hz / (sampling_freq / 2)
    b, a = butter(N=1, Wn=cutoff_nondim, btype="low")
    zi = lfiltic(b, a, x[:1], x_emph[:1])  # initial conditions
    x_d, _ = lfilter(b, a, x_emph, zi=zi)  # deconvoluted
    return x_d


def correction_frequency_response_bilinear(
    freq: Float[ndarray, "time frequency"], Fs: float
):
    assert len(freq.shape) == 2
    assert freq.shape[0] == 1
    assert freq.shape[1] >= 1

    diff_gain = 1.5

    [b, a] = butter(
        1, 1 / (2 * np.pi * diff_gain * Fs / 2)
    )  #  The LP-filter that was applied
    w, junk = freqz(
        b, a, freq.shape[1], fs=Fs, include_nyquist=True
    )  # axis=1 indexes frequencies
    junk = np.absolute(junk) ** 2  #  The mag-squared of the applied LP-filter.
    H = 1 / (1 + (2 * np.pi * freq * diff_gain) ** 2)  #  What should have been applied

    assert np.all(np.allclose(w[newaxis, :], freq))
    bl_correction = H / junk  #  The bilinear transformation correction.
    return bl_correction


def correction_frequency_response_vachon_lueck(
    freq: Float[ndarray, "1 frequency"], pspd: Float[ndarray, "time 1"]
) -> Float[ndarray, "time frequency"]:
    F_0 = 25 * np.sqrt(np.abs(pspd))  # cutoff freq
    tau_therm = 1 / ((2 * np.pi * F_0) / np.sqrt(np.sqrt(2) - 1))  # time constant
    Hinv = (
        1 + (2 * np.pi * tau_therm * freq) ** 2
    ) ** 2  # inverse of the frequency response
    # - correction (Hinv is nondimensional so can apply directly to Pk_gradT)
    return Hinv


def correction_frequency_response():
    return 1


def _try_sigma_derivative(df):
    """
    Cheap try to get sigma0 derivative from high precision cond/temp"""
    from functools import partial

    import gsw
    from level1 import get_vsink
    from util import butterfilt, fft_grad

    def derivative(func, arg, x, dx):
        return (func(**{arg: x + dx}) - func(**{arg: x - dx})) / 2 / dx

    def sigma0_from_C(C, t, p):
        SP = gsw.SP_from_C(C, t, p)
        SA = gsw.SA_from_SP(SP, p, 20, 80)  # lon/lat...
        CT = gsw.CT_from_t(SA, t, p)
        sigma = gsw.sigma0(SA, CT)
        return sigma

    df["dsigmadC"] = np.array(
        [
            derivative(partial(sigma0_from_C, t=t, p=p), "C", C, 0.01)
            for C, t, p in zip(df.Cond, df.Temp, df.Press)
        ]
    )
    df["dsigmadT"] = np.array(
        [
            derivative(partial(sigma0_from_C, C=C, p=p), "t", t, 0.01)
            for C, t, p in zip(df.Cond, df.Temp, df.Press)
        ]
    )

    pspd = get_vsink(data_arrays["PRESSURE"], 1024)[0]
    df["dCdt"] = fft_grad(df.Cmac, 1 / 1024)
    df["dCdt"] = butterfilt(df.dCdt, 50, 1024)
    df["dTdt"] = fft_grad(df.NTCHP, 1 / 1024)
    df["dsigmadt"] = df["dCdt"] * df["dsigmadC"] + df["dTdt"] * df["dsigmadT"]
    df["dsigmadz"] = df.dsigmadt / pspd


def _k_batchelor_mle_round_2():
    """# MLE fitting, round 2
    # This may give
    # TODO: speed up
    range_ = kmax_1 - kmin_1
    allrows = range(cfunc_values_1.shape[0])
    # cost function values to the left, right, and at its maximum value
    _left = cfunc_values_1[allrows, np.maximum(k_batchelor_idx_1 - 1, 0)]
    _right = cfunc_values_1[
        allrows, np.minimum(k_batchelor_idx_1 + 1, cfunc_values_1.shape[0] - 1)
    ]
    _mid = cfunc_values_1[allrows, k_batchelor_idx_1]
    assert np.max(cfunc_values_1[0]) == _mid[0]
    # curvature of cost function values at batchelor wavenumber
    delta_k = (2 * range_) / np.sqrt(2 * _mid - _left - _right)
    print(delta_k)
    finalrange = np.maximum(delta_k, range_)
    # print(finalrange)
    kmin_2: Float[ndarray, "time_slow"] = np.maximum(
        0, k_batchelor_estimates_1 - finalrange
    )
    kmax_2: Float[ndarray, "time_slow"] = k_batchelor_estimates_1 + finalrange
    print(kmin_2, kmax_2)

    k_batchelor_estimates_2 = np.nan * np.zeros_like(chi)
    cfunc_values_2: Float[ndarray, "time_slow k_test"] = np.zeros(
        (chi.shape[0], 30), dtype=float
    )
    for i, (mi, ma) in enumerate(zip(kmin_2, kmax_2)):
        k_test_2 = np.linspace(mi, ma, 30)  # k_batchelor values to test
        print(k_test_2)
        cfunc_values_2[i, :] = get_k_batchelor_costfunc(
            k_test_2,
            chi[i],
            k[i, :],
            Pk[i, :],
            Pnoise,
            viscosity_kinematic,
            diffusivity_temp,
        )
        idx = np.argmax(cfunc_values_2[i, :])
        k_batchelor_estimates_2[i] = k_test_2[idx]
    """
