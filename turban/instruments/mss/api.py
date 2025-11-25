import logging
from typing import cast
from pathlib import Path
import numpy as np

from . import mss_mrd

from turban.instruments.mss.config import MssDeviceConfig
from turban.process.shear.config import ShearConfig
from turban.process.shear.api import ShearLevel1

# Setup logging module
# TODO should handle this more gracefully, having debug level logging everywhere is annoying
# logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
logger = logging.getLogger("turban.instruments.mss")
# logger.setLevel(logging.DEBUG)



def mrd_to_shear_level1(
    fname: str,
    shear_config: ShearConfig,
    mss_config: MssDeviceConfig | None = None,
    shear_sensitivities: dict[str, float] | None = None,
):
    """
    Read a binary .MRD file and convert to level1.

    If no MssDeviceConfig object is given, will read from MRD header. In this case,
    shear_sensitivities must not be None. 
    """
    if mss_config is None:
        # in this case, shear_sensitivities must not be None
        shear_sensitivities = cast(dict[str, float], shear_sensitivities)
        mss_config = MssDeviceConfig.from_mrd(
            filename=fname,
            shear_sensitivities=shear_sensitivities,
            offset=0,
        )

    with open(fname, "rb") as f:
        data_raw = mss_mrd.read_mrd(f)

    data_level0 = mss_mrd.raw_to_level0(mss_config, data_raw)
    data_level1 = mss_mrd.level0_to_level1(mss_config, data_level0)

    sl1 = ShearLevel1(
        time=np.asarray(data_level1["time_count"]),
        senspeed=np.asarray(data_level1["PSPD_REL"]),
        shear=np.asarray(data_level1["SHEAR"]),
        section_number=np.ones_like(data_level1["time_count"], dtype=int),
        cfg=shear_config,
    )

    return sl1

