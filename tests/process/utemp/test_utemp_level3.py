from pathlib import Path
import numpy as np
from turban.process.utemp.api import UTempLevel1, UTempProcessing
from turban.process.utemp.config import FP07Response, UTempConfig
from turban.process.utemp.level3 import get_noise_empirical

top_level = Path(__file__).resolve().parent.parent.parent.parent


def test_get_noise():
    x = np.arange(24, dtype=float).reshape(2, 3, 4)
    y = get_noise_empirical(x)
    assert y.shape == (2, 4)


def test_custom_fp07_response_is_applied():
    """A custom FP07 response must not be overridden by a preset of the same name."""
    n = 20 * 512
    rng = np.random.default_rng(0)
    dtempdt = 0.01 * rng.standard_normal((2, n))

    def psi_k(fp07_response):
        cfg = UTempConfig(
            sampfreq=512.0,
            segment_length=256,
            segment_overlap=128,
            chunk_length=1024,
            chunk_overlap=512,
            fp07_response=fp07_response,
        )
        level1 = UTempLevel1(
            time=np.arange(n) / 512.0,
            senspeed=0.7 * np.ones(n),
            dtempdt=dtempdt,
            section_number=np.ones(n, dtype=int),
            cfg=cfg,
        )
        return UTempProcessing(level1).level3.psi_k_gradt

    # same parameters as the "ijichi25" preset, but labelled "nash99"
    relabelled = FP07Response(name="nash99", tau0=5e-3, gamma=-0.5, npoles=1)
    np.testing.assert_allclose(psi_k(relabelled), psi_k("ijichi25"))
    assert not np.allclose(psi_k(relabelled), psi_k("nash99"))
