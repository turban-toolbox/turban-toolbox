import pytest
import numpy as np
from turban.process.shear.level3 import power_spectrum
from turban.utils.util import agg_fast_to_slow, reshape_overlap_index


def test_spectra_arr_shape():
    """
    For each channel: Need as many spectra as average values
    """
    N = 20
    x = np.repeat(np.arange(N).astype(float)[np.newaxis, ...], 3, axis=0)
    assert x.shape == (3, N)
    fft_length = 6
    fft_overlap = 3
    sampling_freq = 1.0
    diss_length = 12
    diss_overlap = 9

    y1, f = power_spectrum(
        x, sampling_freq, fft_length, fft_overlap, diss_length, diss_overlap
    )
    # platform speed
    y2 = agg_fast_to_slow(
        x,
        data_len=N,
        fft_length=fft_length,
        fft_overlap=fft_overlap,
        diss_length=diss_length,
        diss_overlap=diss_overlap,
    )

    assert f.shape == (fft_length / 2 + 1,), "Wrong number of frequencies"
    assert y1.shape[:2] == y2.shape
    assert y1.shape[-1] == len(f)
