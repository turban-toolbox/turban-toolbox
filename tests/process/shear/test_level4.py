import pytest
import numpy as np
from numpy import newaxis
import matplotlib.pyplot as plt

from turban.process.shear.level4 import (
    unwrap_base2,
    model_spectrum,
    model_spectrum_lueck,
)


def test_unwrap_unwrap_base2():
    q = np.array([0, 1, 2, 3, 9])
    qd = unwrap_base2(q)
    assert np.all(qd[1] == np.array([False, True, False, True, True]))
    assert np.all(qd[2] == np.array([False, False, True, True, False]))
    assert np.all(qd[4] == np.array([False, False, False, False, False]))
    assert np.all(qd[8] == np.array([False, False, False, False, True]))
    assert np.all(qd[16] == np.array([False, False, False, False, False]))


def test_model_spectrum():
    waveno = np.linspace(0, 100)[newaxis, :]
    psi = model_spectrum(
        waveno=waveno,
        eps=np.array([1.1e-7]),
        molvisc=np.array([1.6e-6]),
    )
    fig = plt.figure(figsize=(9, 9))
    ax = fig.subplots()
    ax.loglog(waveno.T, psi.T)
    ax.set_xlabel("Wavenumber")
    ax.set_ylabel("Power spectral density")
    ax.legend(["Lueck 2022"])
    ax.grid()
    ax.set_title("Cf. ATOMIX paper Fig. 12")
    fig.savefig(f"out/tests/test_model_spectrum.png")


def test_model_spectrum_nondim():
    kn = 10 ** np.linspace(np.log10(3e-4), np.log10(0.3))
    psin = model_spectrum_lueck(kn)
    fig = plt.figure(figsize=(9, 9))
    ax = fig.subplots()
    ax.loglog(kn, psin)
    ax.set_xlabel("Nondimensional wavenumber")
    ax.set_ylabel("Nondimensional power spectral density")
    ax.legend(["Lueck 2022"])
    ax.grid()
    ax.set_ylim([1e-2, 2.5])
    ax.set_title("Cf. ATOMIX paper Fig. 3")
    fig.savefig(f"out/tests/test_model_spectrum_nondim.png")
