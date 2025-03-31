import sys
from pathlib import Path
import numpy as np

top_level = Path(__file__).resolve().parent.parent.parent.parent

sys.path.insert(0, str(top_level))

from turban.instruments.mss import MSS
from turban.shear import ShearConfig


def test_mss():
    mss = MSS()

    mss.read_mrd(top_level / "data" / "mss" / "Nien0020.MRD")

    shear_config = ShearConfig(
        sampling_freq=mss.cfg.sampling_freq,
        fft_length=2048,
        fft_overlap=1024,
        diss_length=4 * 2048,
        diss_overlap=1024,
    )
    p = mss.to_shear_processing(
        section_marker=np.ones_like(mss.mrd.level0["PRESS"], dtype=int),
        cfg=shear_config,
    )
    assert isinstance(p.level4.eps, np.ndarray)
