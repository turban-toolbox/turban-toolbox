import numpy as np
import pytest

from turban.process.utemp.level4 import (
    eps_from_kbatch,
    temperature_dissipation,
    theoretical_spectrum,
)

MOLVISC, MOLDIFF = 1.6e-6, 1.4e-7


def _synthetic(model, q, kbatch, chi, noise, ntime=3):
    freq = np.fft.rfftfreq(2048, 1 / 1024)
    senspeed = np.linspace(0.6, 1.0, ntime)
    waveno = freq[np.newaxis, :] / senspeed[:, np.newaxis]
    psi_noise = np.full((1, len(freq)), noise)
    psi = theoretical_spectrum(model, waveno, kbatch, chi, MOLDIFF, q)
    return waveno, psi[np.newaxis] + psi_noise[:, np.newaxis, :], psi_noise


def _run(waveno, psi, psi_noise, model, q):
    ntime = waveno.shape[0]
    return temperature_dissipation(
        psi_k_gradt=psi,
        waveno=waveno,
        psi_noise=psi_noise,
        waveno_limit_upper=np.inf,
        molvisc=np.full(ntime, MOLVISC),
        moldiff=np.full(ntime, MOLDIFF),
        spectrum_model=model,
        q=q,
        n_trial=100,
        dof=6.0,
    )


@pytest.mark.parametrize("model,q", [("batchelor", 3.7), ("kraichnan", 5.26)])
def test_temperature_dissipation_synthetic(model, q):
    kbatch, chi = 150.0, 1e-8
    waveno, psi, psi_noise = _synthetic(model, q, kbatch, chi, 1e-7)
    chi_est, eps, eps_lower, eps_upper, kb, kb_ci, *_ = _run(
        waveno, psi, psi_noise, model, q
    )
    assert np.allclose(kb, kbatch, rtol=0.03)
    assert np.allclose(chi_est, chi, rtol=0.1)
    eps_true = eps_from_kbatch(np.array([[kbatch]]), MOLVISC, MOLDIFF)
    assert np.allclose(eps, eps_true, rtol=0.15)
    assert np.all((eps_lower <= eps) & (eps <= eps_upper))


def test_temperature_dissipation_noise_only():
    waveno, _, psi_noise = _synthetic("batchelor", 3.7, 150.0, 1e-8, 1e-7)
    psi = np.broadcast_to(psi_noise[:, np.newaxis, :], (1, *waveno.shape)).copy()
    out = _run(waveno, psi, psi_noise, "batchelor", 3.7)
    assert all(np.all(np.isnan(x)) for x in out)
