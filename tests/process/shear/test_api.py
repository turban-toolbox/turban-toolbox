import numpy as np

from turban import ShearConfig, ShearLevel1, ShearProcessing
from turban.variables import VARIABLES
from turban.process.shear.variables import VARIABLES_SHEAR


def test_default_vocabulary_in_xarray(check_vocabulary_in_xarray):
    n = 20 * 512
    rng = np.random.default_rng(0)
    level1 = ShearLevel1(
        time=np.arange(n) / 512.0,
        senspeed=0.7 * np.ones(n),
        shear=0.1 * rng.standard_normal((2, n)),
        section_number=np.ones(n, dtype=int),
        cfg=ShearConfig(
            sampfreq=512.0,
            segment_length=256,
            segment_overlap=128,
            chunk_length=1024,
            chunk_overlap=512,
        ),
    )
    check_vocabulary_in_xarray(ShearProcessing(level1), [VARIABLES, VARIABLES_SHEAR])
