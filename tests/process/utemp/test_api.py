import numpy as np

from turban.process.utemp.api import UTempLevel1, UTempProcessing
from turban.process.utemp.config import UTempConfig
from turban.variables import VARIABLES
from turban.process.utemp.variables import VARIABLES_UTEMP


def test_default_vocabulary_in_xarray(check_vocabulary_in_xarray):
    n = 20 * 512
    rng = np.random.default_rng(0)
    level1 = UTempLevel1(
        time=np.arange(n) / 512.0,
        senspeed=0.7 * np.ones(n),
        dtempdt=0.01 * rng.standard_normal((2, n)),
        section_number=np.ones(n, dtype=int),
        cfg=UTempConfig(
            sampfreq=512.0,
            segment_length=256,
            segment_overlap=128,
            chunk_length=1024,
            chunk_overlap=512,
            fp07_response="nash99",
        ),
    )
    check_vocabulary_in_xarray(UTempProcessing(level1), [VARIABLES, VARIABLES_UTEMP])
