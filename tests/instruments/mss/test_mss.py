import numpy as np
from tests.fixtures import mss_mrd_filename
from turban.instruments.mss import MSS
from turban.shear import ShearConfig


def test_mss(mss_mrd_filename):
    mss = MSS()

    mss.read_mrd(mss_mrd_filename)

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
