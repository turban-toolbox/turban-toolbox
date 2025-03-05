#!/usr/bin/env python

import math
import numba as nb

from numba import jit
import numpy as np
from numpy import ndarray


@jit
def exceeds_until(x, y):
    assert len(x) == len(y)
    (inds,) = np.where(x >= y)
    if len(inds):
        return inds[-1]
    else:
        return len(x)


@jit
def integrate(
    y,
    x,
    x_from,
    x_to,
):
    (ii,) = np.where((x_from <= x) & (x <= x_to))
    return np.trapz(y[ii], x=x[ii])


@jit
def chisquared(x, dof):
    """
    pdf of chi^2 distribution
    """
    return x ** (dof / 2) * np.exp(-x / 2) / (2 ** (dof / 2) * math.gamma(dof / 2))


@jit
def costfunction_c11(psi: ndarray, psi_theoretical: ndarray) -> float:
    dof = 6.0
    # degrees of freedom
    cost_vector = chisquared(dof * psi / psi_theoretical, dof)
    return np.nanmean(cost_vector)


@jit
def get_k_batchelor_estimate(
    k_test, chi, wavenumber, psi, psi_noise, viscosity_kinematic, diffusivity_temp
):
    # start value
    values = np.zeros_like(k_test)
    for i, k_batchelor in enumerate(k_test):
        # loop over wavenumbers
        psi_theoretical = (
            theoretical_spectrum(
                wavenumber,
                k_batchelor,
                chi,
                viscosity_kinematic,
                diffusivity_temp,
            )
            + psi_noise
        )
        values[i] = costfunction_c11(psi, psi_theoretical)

    return np.nanmean(k_test[values == np.nanmax(values)])


@jit
def theoretical_spectrum(
    wavenumber: ndarray,
    k_batchelor: float,
    chi: float,
    viscosity_kinematic: float,
    diffusivity_temp: float,
) -> ndarray:
    # TODO k in radian units
    # 1D temperature gradient spectrum (Batchelor)
    # Eqn. 7 in Peterson et al. 2014

    qB = 3.7
    eps = k_batchelor ** (4.0) * viscosity_kinematic * diffusivity_temp ** (2.0)

    # temp. gradient spectrum
    spec = np.zeros_like(wavenumber)
    for i, k in enumerate(wavenumber):
        spec[i] = (
            chi
            * np.sqrt(viscosity_kinematic / eps)
            / k_batchelor
            * (
                k ** (2.0)
                * (
                    qB
                    * k_batchelor
                    / k
                    * np.exp(-qB * k ** (2.0) / k_batchelor ** (2.0))
                    + np.sqrt(np.pi)
                    * qB ** (1.5)
                    * (math.erf(np.sqrt(qB) * k / k_batchelor) - 1.0)
                )
            )
        )
    return spec


@jit
def temperature_dissipation(
    wavenumber: ndarray[float],
    psi: ndarray[float],
    psi_noise: ndarray[float],
    wavenumber_limit_upper: float,
):
    """
    pub fn temperature_diss(
            wavenumber: &[f64],
            psi: &[f64],
            psi_noise: &[f64],
            platform_speed: f64,
            wavenumber_limit_upper: f64,
    ) -> Result<f64, f64> {
    """
    #   nu, kin. viscosity of water; assumed known constant
    viscosity_kinematic = 0.0000016
    # molecular temperature diffusivity [m^2/s]
    diffusivity_temp = 0.00000014

    # upper limit
    # find last point where spec exceeds 2x noise spec

    idx_above_noise = exceeds_until(psi, 2 * psi_noise)
    ku = min(wavenumber[idx_above_noise], wavenumber_limit_upper)

    # subtract noise

    psi = psi - psi_noise

    # integrate spectrum, 1st pass
    _spec_int = integrate(psi, wavenumber, wavenumber[1], ku)
    if _spec_int < 0.0:
        _spec_int = integrate(psi_noise, wavenumber, wavenumber[1], ku)
    chi = 6.0 * diffusivity_temp * _spec_int

    kmax = 3.0 * np.nanmax(wavenumber)
    kmin = kmax / 600.0
    k_test = np.linspace(kmin, kmax, 30)  # k_batchelor values to test

    k_batchelor_estimate = get_k_batchelor_estimate(
        k_test, chi, wavenumber, psi, psi_noise, viscosity_kinematic, diffusivity_temp
    )
    return chi, k_batchelor_estimate


@jit
def detrend(x: ndarray):
    return x - np.mean(x)


@jit
def apply_cosine_window(x: ndarray):
    return x * np.hanning(len(x))


@jit
def mean_axis0(arr: ndarray):
    return np.array([arr[:, i].mean() for i in range(arr.shape[1])])


@jit
def get_noise(spectra: ndarray) -> ndarray:
    """
    Spectra - 1st dimension is index of spectrum
    2nd dimension is frequency
    """
    assert spectra.ndim == 2
    # a measure of the intensity of the spectrum
    with nb.objmode(spec_intens="float64[:]"):
        spec_intens = np.mean(spectra[:, :20], axis=1)
    # 5 % least intense spectra
    (ii,) = np.where(spec_intens < np.percentile(spec_intens, 5))
    noise = 10 ** mean_axis0(np.log10(spectra[ii, :]))
    return noise


@jit
def power_spectrum(x: ndarray, fft_length: int, sampling_freq: float):
    assert fft_length % 2 == 0
    data_length = len(x)
    # half-overlapping windows
    N = 2 * (data_length // fft_length) - 1
    # collect power density spectra
    spectra = np.zeros((N, int(fft_length / 2 + 1)))
    for i in range(N):
        i0 = i * int(fft_length / 2)
        i1 = i0 + fft_length
        xc = x[i0:i1]
        xc = detrend(xc)
        xc = apply_cosine_window(xc)
        with nb.objmode(F=float64[:]):
            F = np.fft.rfft(xc)
        spectra[i, :] = (F.conj() * F).real

    return mean_axis0(spectra) / fft_length / (sampling_freq / 2)


from numba import float64, int32
@jit(float64[:](float64[:],float64,int32,int32,float64))
def process_temperature_series(
    x: ndarray[float],
    # platform_speed: ndarray,
    wavenumber_limit_upper: float,
    chunk_length: int,
    fft_length: int,
    sampling_freq: float,
):
    assert chunk_length % 2 == 0
    assert x.ndim == 1
    # number of half-overlapping windows
    N = 2 * (len(x) // chunk_length) - 1
    spec_length = int(fft_length / 2 + 1)
    spectra = np.zeros((N, spec_length))
    pspd = np.zeros(N)
    for i in range(N):
        i0 = i * int(chunk_length / 2)
        i1 = i0 + chunk_length
        xc = x[i0:i1]
        spectra[i, :] = power_spectrum(xc, fft_length, sampling_freq)
        # pspd[i] = np.mean(platform_speed[i0:i1])

    Pnoise = get_noise(spectra)
    with nb.objmode(freq="float64[:]"):
        freq = np.fft.rfftfreq(fft_length)

    chi = np.zeros(N)
    k_batchelor = np.zeros(N)
    for i, P in enumerate(spectra):
        pspd = 1  # TODO
        k = freq / pspd
        chi[i], k_batchelor[i] = temperature_dissipation(
            wavenumber=k,
            psi=P * pspd,
            psi_noise=Pnoise,
            wavenumber_limit_upper=wavenumber_limit_upper,
        )

    return chi


if __name__ == "__main__":
    # chi = temperature_dissipation(wavenumber, psi, psi_noise, 1.0, 10000.0)
    # print(chi)
    x = np.random.rand(int(1e5))
    res = process_temperature_series(
        x,
        wavenumber_limit_upper=1e3,
        chunk_length=5024,
        fft_length=1024,
        sampling_freq=1 / 1024.0,
    )
    print(res)
