import pytest
import numpy as np
from turban.process.shear.level3 import spectrum, apply_compensation_highpass
from turban.utils.util import agg_fast_to_slow, reshape_overlap_index, butterfilt


@pytest.mark.parametrize("freq_highpass", [0.1, 2.0])
def test_apply_compensation_highpass_restores_spectrum(freq_highpass):
    """
    High-pass filter white noise as in level 2, then check that the level 3
    compensation recovers the spectrum of the unfiltered signal.
    """
    rng = np.random.default_rng(42)
    sampfreq = 64.0
    segment_length = 256  # tau_fft = 4 s, lowest non-zero freq 0.25 Hz
    n = 64 * segment_length * 4
    x = rng.standard_normal((2, n))
    specarg = dict(
        sampfreq=sampfreq,
        segment_length=segment_length,
        segment_overlap=segment_length // 2,
        chunk_length=4 * segment_length,
        chunk_overlap=0,
        section_number=np.ones(n, dtype=int),
    )

    psi_true, freq = spectrum(x, **specarg)
    # filter in same way as in level 2
    x_hp = butterfilt(
        signal=x, cutoff_freq_Hz=freq_highpass, sampfreq=sampfreq, btype="high"
    )
    psi_hp, _ = spectrum(x_hp, **specarg)

    # Compare where the filter attenuation is moderate (f >= 2 f_HP, correction
    # <= (5/4)^2): below that, the large correction factor amplifies spectral
    # leakage.
    mask = freq >= 2 * freq_highpass
    ratio_uncorrected = (psi_hp / psi_true)[..., mask].mean(axis=(0, 1))

    _ = apply_compensation_highpass(psi_hp[..., 1:], freq[1:], freq_highpass)
    ratio = (psi_hp / psi_true)[..., mask].mean(axis=(0, 1))

    # filtering visibly reduces the lowest compared frequencies ...
    assert ratio_uncorrected[0] < 0.85
    # ... and the compensation restores them. The lowest non-zero bin is also
    # biased by windowing and detrending, so it gets a looser tolerance.
    is_lowest = freq[mask] == freq[1]
    np.testing.assert_allclose(ratio[is_lowest], 1.0, rtol=0.1)
    np.testing.assert_allclose(ratio[~is_lowest], 1.0, rtol=0.05)


def test_spectra_arr_shape():
    """
    For each channel: Need as many spectra as average values
    """
    N = 20
    x = np.repeat(np.arange(N).astype(float)[np.newaxis, ...], 3, axis=0)
    assert x.shape == (3, N)
    segment_length = 6
    segment_overlap = 3
    sampfreq = 1.0
    chunk_length = 12
    chunk_overlap = 9

    y1, f = spectrum(
        x,
        sampfreq,
        segment_length,
        segment_overlap,
        chunk_length,
        chunk_overlap,
        section_number=np.ones(N, dtype=int),
    )
    # platform speed
    y2 = agg_fast_to_slow(
        x,
        section_number_or_data_len=N,
        chunk_length=chunk_length,
        chunk_overlap=chunk_overlap,
    )

    assert f.shape == (segment_length / 2 + 1,), "Wrong number of frequencies"
    assert y1.shape[:2] == y2.shape
    assert y1.shape[-1] == len(f)
