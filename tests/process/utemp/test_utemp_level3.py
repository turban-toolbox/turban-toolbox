from pathlib import Path
import numpy as np
from turban.process.utemp.level3 import get_noise

top_level = Path(__file__).resolve().parent.parent.parent.parent

def test_get_noise():
    x = np.arange(24, dtype=float).reshape(2, 3, 4)
    y = get_noise(x)
    assert y.shape == (2, 4)