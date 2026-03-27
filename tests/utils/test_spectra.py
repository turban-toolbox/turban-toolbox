import pytest
import xarray as xr
from matplotlib import pyplot as plt
import numpy as np
from turban.utils.util import get_chunking_index
from turban.utils.spectra import remove_vibration_goodman, spectrum

from tests.filepaths import atomix_benchmark_faroe_fpath


@pytest.mark.parametrize("phase_deg", [0, 90, 180, 270])
def test_remove_vibration_goodman_synthetic(phase_deg):
    """Signal is the sum of two sine curves; both are provided as vibration references.

    After Goodman removal the cleaned PSD must be close to zero because the signal
    carries no variance independent of the two vibration channels.
    Phase offset (degrees) is applied to the second vibration channel.
    """
    sampfreq = 512.0
    n_samples = 500_000
    t = np.arange(n_samples) / sampfreq

    f1, A1 = 5.0, 1.0  # Hz, amplitude
    f2, A2 = 43.0, 0.5
    phase_rad = np.deg2rad(phase_deg)
    vib1 = A1 * np.sin(2 * np.pi * f1 * t)
    vib2 = A2 * np.sin(2 * np.pi * f2 * t + phase_rad)
    signal = vib1 + vib2  # signal IS the sum → no independent variance

    # shapes expected by remove_vibration_goodman: (nsig, time) and (nvib, time)
    signal_2d = signal[np.newaxis, :]  # (1, N)
    vib_2d = np.stack([vib1, vib2], axis=0)  # (2, N)

    chunk_length = n_samples
    chunk_overlap = 0
    segment_length = 1024
    segment_overlap = 512

    reshape_index = get_chunking_index(n_samples, (chunk_length, chunk_overlap))

    specarg = dict(
        sampfreq=sampfreq,
        reshape_index=reshape_index,
        segment_length=segment_length,
        segment_overlap=segment_overlap,
    )

    psi_f_cleaned, freq, psi_f_uncl = remove_vibration_goodman(
        signal_2d, vib_2d, **specarg
    )

    # diagonal element [0, 0]: shape (nchunk, nfreq)
    cleaned_diag = psi_f_cleaned[0, 0].real
    uncleaned_diag = psi_f_uncl[0, 0].real

    # skip DC bin; cleaned power should be negligible relative to uncleaned
    ratio = np.abs(cleaned_diag[:, 1:]) / np.abs(uncleaned_diag[:, 1:])
    print(ratio)
    assert np.all(
        ratio < 1e-4
    ), f"Cleaned spectrum not close to zero: max ratio = {ratio.max():.2e}"

    # # --- individual PSDs of each vibration signal ---
    # psi_vib1, _ = spectrum(vib1[np.newaxis, :], **specarg)  # (1, nchunk, nfreq)
    # psi_vib2, _ = spectrum(vib2[np.newaxis, :], **specarg)

    # # --- plot ---
    # fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # # top: timeseries (first second)
    # n_show = int(sampfreq)
    # ax = axes[0]
    # ax.plot(t[:n_show], signal[:n_show], label="signal = vib1 + vib2", lw=1)
    # ax.plot(t[:n_show], vib1[:n_show], label=f"vib1  ({f1} Hz)", lw=1, ls="--")
    # ax.plot(t[:n_show], vib2[:n_show], label=f"vib2  ({f2} Hz)", lw=1, ls=":")
    # ax.set_xlabel("Time (s)")
    # ax.set_ylabel("Amplitude")
    # ax.set_title("Timeseries (first second)")
    # ax.legend(fontsize=8)

    # # bottom: power spectral densities
    # ax = axes[1]
    # ax.semilogy(freq[1:], uncleaned_diag.mean(axis=0)[1:], label="Signal uncleaned")
    # ax.semilogy(freq[1:], np.abs(cleaned_diag.mean(axis=0))[1:], label="Signal cleaned", ls="--")
    # ax.semilogy(freq[1:], psi_vib1[0].real.mean(axis=0)[1:], label=f"PSD vib1 ({f1} Hz)", ls="-.", alpha=0.8)
    # ax.semilogy(freq[1:], psi_vib2[0].real.mean(axis=0)[1:], label=f"PSD vib2 ({f2} Hz)", ls="dotted", alpha=0.8)
    # ax.set_xlabel("Frequency (Hz)")
    # ax.set_ylabel("PSD")
    # ax.set_title("Power spectral densities")
    # ax.legend(fontsize=8)

    # fig.tight_layout()
    # fig.savefig("out/tests/utils/goodman_synthetic.png")
    # plt.close(fig)
