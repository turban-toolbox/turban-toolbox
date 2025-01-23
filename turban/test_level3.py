import pytest
import numpy as np
from atomixpy.level3 import spectra
from atomixpy.util import average_fast_to_slow


def test_spectra_arr_shape():
    """
    For each channel: Need as many spectra as average values
    """
    N = 20
    x = np.repeat(np.arange(N).astype(float)[np.newaxis, ...], 3, axis=0)
    assert x.shape == (3, N)
    fftlen = 6
    sampling_freq = 1.0
    chunklen = 3
    chunkoverlap = 2

    y1, f = spectra(x, fftlen, sampling_freq, chunklen, chunkoverlap)
    # platform speed
    y2 = average_fast_to_slow(x, fftlen, chunklen, chunkoverlap)

    assert f.shape == (fftlen / 2 + 1,), "Wrong number of frequencies"
    assert y1.shape[:2] == y2.shape
    assert y1.shape[-1] == len(f)
