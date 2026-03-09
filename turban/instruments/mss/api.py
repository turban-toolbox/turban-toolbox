from typing import cast, Literal
from pathlib import Path
import numpy as np

from turban.process.utemp.api import UTempLevel1
from turban.process.utemp.config import UTempConfig

from turban.instruments.mss import mss_mrd

from turban.instruments.mss.config import MssDeviceConfig
from turban.process.shear.config import ShearConfig
from turban.process.shear.api import ShearLevel1
from turban.utils.logging import get_logger

logger = get_logger(__name__)


def mrd_to_level1(
    fname: str | Path,
    target: Literal["shear", "utemp"],
    proc_cfg: ShearConfig | UTempConfig,
    mss_cfg: MssDeviceConfig | None = None,
    shear_sensitivities: dict[str, float] | None = None,
) -> ShearLevel1 | UTempLevel1:
    """
    Read a binary .MRD file and convert to level1.

    If no MssDeviceConfig object is given, will read from MRD header. In this case,
    shear_sensitivities must not be None.
    """
    if mss_cfg is None:
        # in this case, shear_sensitivities must not be None
        shear_sensitivities = cast(dict[str, float], shear_sensitivities)
        mss_cfg = MssDeviceConfig.from_mrd(
            filename=fname,
            shear_sensitivities=shear_sensitivities,
            offset=0,
        )

    with open(fname, "rb") as f:
        data_raw = mss_mrd.read_mrd(f)

    data_level0 = mss_mrd.raw_to_level0(mss_cfg, data_raw)
    data_level1 = mss_mrd.level0_to_level1(mss_cfg, data_level0)

    if target == "shear":
        level1 = ShearLevel1(
            time=np.asarray(data_level1["time_count"]),
            senspeed=np.asarray(data_level1["PSPD_REL"]),
            shear=np.asarray(data_level1["SHEAR"]),
            section_number=np.ones_like(data_level1["time_count"], dtype=int),
            cfg=cast(ShearConfig, proc_cfg),
        )

    elif target == "utemp":
        level1 = UTempLevel1(
            time=np.asarray(data_level1["time_count"]),
            senspeed=np.asarray(data_level1["PSPD_REL"]),
            dtempdt=np.asarray(data_level1["utemp"]),
            section_number=np.ones_like(data_level1["time_count"], dtype=int),
            cfg=cast(UTempConfig, proc_cfg),
        )

    for varname in ["SA", "CT", "Press", "DENS"]:
        level1.add_aux_data(data_level1[varname].values, varname)

    return level1
