import pytest
import numpy as np

from turban.shear.level4 import unwrap_quality_metric


def test_unwrap_quality_metric():
    q = np.array([0, 1, 2, 3, 9])
    qd = unwrap_quality_metric(q)
    assert np.all(qd[1] == np.array([False, True, False, True, True]))
    assert np.all(qd[2] == np.array([False, False, True, True, False]))
    assert np.all(qd[4] == np.array([False, False, False, False, False]))
    assert np.all(qd[8] == np.array([False, False, False, False, True]))
    assert np.all(qd[16] == np.array([False, False, False, False, False]))
