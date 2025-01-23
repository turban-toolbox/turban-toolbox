import pytest
import numpy as np
from .util import (
    fft_grad,
    _reshape_any_inds,
    reshape_any_first,
    reshape_any_nextlast,
    reshape_any_last,
    reshape_halfoverlap_first,
    reshape_halfoverlap_last,
    average_fast_to_slow,
    binned_gradient_halfoverlap,
)


def test_fft_grad():
    """Hard to test without hardcoding finicky values...
    let's make something that at least checks plausibility
    """
    x = np.arange(1000, dtype=float)
    y = fft_grad(x, 0.1)
    assert 9 < np.mean(y) < 11


def test_reshape():

    for N in range(25, 50):
        x1 = np.arange(N).astype(float)
        x2 = reshape_halfoverlap_last(x1, 4)
        y1 = reshape_halfoverlap_first(x1, 12)
        y2 = reshape_any_first(x2, 5, 2)
        assert len(y1) == len(y2), f"Failed for N={N}"


def test_reshape_halfoverlap_samedim():
    """ """
    for N in range(10, 50):
        x = np.arange(N).astype(float)
        y1 = reshape_halfoverlap_last(x, 6)
        y2 = reshape_halfoverlap_first(x, 6)
        assert len(y1) == len(y2), f"Failed for N={N}"


def test_reshape_halfoverlap_first():
    x = np.ones((10, 3))
    assert reshape_halfoverlap_first(x, 4).shape == (4, 4, 3)
    assert reshape_halfoverlap_first(x, 6).shape == (2, 6, 3)
    with pytest.raises(Exception):
        reshape_halfoverlap_first(x, 5)


def test_reshape_halfoverlap_last():
    x = np.ones((3, 10))
    assert reshape_halfoverlap_last(x, 4).shape == (3, 4, 4)
    assert reshape_halfoverlap_last(x, 6).shape == (3, 2, 6)
    with pytest.raises(Exception):
        reshape_halfoverlap_last(x, 5)


def test_reshape_too_short():
    x = np.arange(10, dtype=float)
    w = 20
    assert reshape_halfoverlap_first(x, w).shape == (0, w)
    assert reshape_halfoverlap_last(x, w).shape == (0, w)

    assert _reshape_any_inds(w, 1, 10) == []
    assert reshape_any_first(x, w, 1).shape == (0, w)
    assert reshape_any_nextlast(np.ones((4, 10, 3)), w, 1).shape == (4, 0, w, 3)
    assert reshape_any_last(x, w, 1).shape == (0, w)


def test_reshape_any_first():
    x = np.ones((10, 3))
    assert reshape_any_first(x, 5, 1).shape == (2, 5, 3)


def test_reshape_any_nextlast():
    assert reshape_any_nextlast(np.ones((2, 10, 3)), 5, 1).shape == (2, 2, 5, 3)


def test_reshape_any_last():
    assert reshape_any_last(np.ones((3, 10)), 5, 1).shape == (3, 2, 5)
    assert reshape_any_last(np.ones((3, 4)), 3, 2).shape == (3, 2, 3)


def test_average_fast_to_slow():
    x = np.repeat(np.arange(10).astype(float)[np.newaxis, ...], 3, axis=0)
    assert x.shape == (3, 10)
    # half-overlapping intervals of length 4: 4 intervals
    # then averaged over intervals of length 3 with overlap 1: 2
    y = average_fast_to_slow(x, 4, 3, 2)
    assert y.shape == (3, 2)


def test_binned_gradient_halfoverlap():
    x = np.arange(10, dtype=float)
    pspd = np.ones_like(x)
    dxdt = binned_gradient_halfoverlap(
        x, pspd, chunklen_samples=4, sampling_frequency=1.0
    )
    expected = np.ones(4, dtype=float)
    assert np.allclose(dxdt, expected, rtol=1e-10)