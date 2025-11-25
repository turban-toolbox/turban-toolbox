import pytest
import numpy as np
from turban.utils.util import (
    fft_grad,
    reshape_overlap_index,
    reshape_any_first,
    reshape_any_nextlast,
    reshape_any_last,
    reshape_halfoverlap_first,
    reshape_halfoverlap_last,
    agg_fast_to_slow,
    get_cleaned_fraction,
    diss_chunk_wise_reshape_index,
    get_chunking_index,
    boolarr_to_sections,
)


def test_get_cleaned_fraction():
    x = np.arange(20)
    xc = x.copy()
    xc[2:4] = 0  # clean two samples in the first diss_chunk of 10 samples
    cl_frac = get_cleaned_fraction(
        x,
        xc,
        section_number_or_data_len=len(x),
        segment_length=4,
        segment_overlap=2,
        chunk_length=10,
        chunk_overlap=0,
    )
    assert np.all(cl_frac == np.array([2 / 10, 0]))


def test_diss_chunk_wise_reshape_index():
    ii = get_chunking_index(
        20,
        (10, 0),
        (4, 2),
    )
    assert ii.shape == (2, 4, 4)
    assert diss_chunk_wise_reshape_index(ii).shape == (2, 10)


def test_boolarr_to_sections():
    bools = np.array([True, False, False, True, True, False, True])
    result = [[0], [3, 4], [6]]
    assert boolarr_to_sections(bools) == result

    bools = np.array([True, False, False, True, True, False])
    result = [[0], [3, 4]]
    assert boolarr_to_sections(bools) == result

    bools = np.array([False, False, True, True, False, True])
    result = [[2, 3], [5]]
    assert boolarr_to_sections(bools) == result

    bools = np.array([False, False, True, True, False])
    result = [[2, 3]]
    assert boolarr_to_sections(bools) == result


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


def test_reshape_any_inds():
    assert np.all(
        reshape_overlap_index(4, 2, 10)
        == np.array(
            [
                [0, 1, 2, 3],
                [2, 3, 4, 5],
                [4, 5, 6, 7],
                [6, 7, 8, 9],
            ]
        )
    )
    assert np.all(
        reshape_overlap_index(6, 3, 10) == reshape_halfoverlap_last(np.arange(10.0), 6)
    )
    assert np.all(reshape_overlap_index(20, 1, 10) == np.zeros((0, 20), dtype=int))


def test_reshape_too_short():
    x = np.arange(10, dtype=float)
    w = 20
    assert reshape_halfoverlap_first(x, w).shape == (0, w)
    assert reshape_halfoverlap_last(x, w).shape == (0, w)

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
    y = agg_fast_to_slow(
        x,
        section_number_or_data_len=x.shape[-1],
        segment_length=4,
        segment_overlap=2,
        chunk_length=6,
        chunk_overlap=2,
    )

    assert y.shape == (3, 2)


def test_agg_fast_to_slow():
    x = np.arange(20.0)
    xm = agg_fast_to_slow(
        x,
        section_number_or_data_len=len(x),
        segment_length=4,
        segment_overlap=2,
        chunk_length=10,
        chunk_overlap=0,
        agg_method="max",
    )
    assert np.all(xm == np.array([9.0, 19.0]))
