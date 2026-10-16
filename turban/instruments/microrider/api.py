from typing import Any, Literal

import numpy as np
from jaxtyping import Float

from turban.instruments.generic.api import Instrument
from turban.instruments.generic.config import InstrumentConfig
from turban.instruments.microrider import rsIO
from turban.instruments.microrider.sensorspeedplugins import (
    senspeed_constant,
    senspeed_from_em,
    senspeed_from_lookup_table,
)
from turban.process.shear.api import ShearLevel1
from turban.process.shear.config import ShearConfig

from turban.utils.logging import get_logger

logger = get_logger(__name__)


class MicroriderAPIError(Exception):
    pass


class MicroriderConfig(InstrumentConfig):
    # Channel settings overriding those of the setup string in the .p file,
    # e.g. [{"name": "sh1", "sens": 0.0712}]
    channel_cfgs: list[dict[str, Any]] = []


class MicroriderProbe(Instrument):

    def __init__(self, cfg: MicroriderConfig) -> None:
        self.cfg: MicroriderConfig = cfg

    def to_shear_level1(
        self,
        filename: str,
        cfg: ShearConfig,
        senspeed: (
            float
            | Literal["em"]
            | tuple[Float[np.ndarray, "n"], Float[np.ndarray, "n"]]
        ),
    ) -> ShearLevel1:
        """Read a MicroRider .p file and convert it to a ShearLevel1 dataclass.

        Parameters
        ----------
        filename : str
            Path to the MicroRider binary .p file.
        cfg : ShearConfig
            Processing configuration for the shear pipeline.
        senspeed : float | "em" | tuple[ndarray, ndarray]
            Source of the sensor speed:

            - ``float``: constant sensor speed in m/s,
            - ``"em"``: interpolated from the ``U_EM`` channel in the file,
            - ``(time, senspeed)``: lookup table, linearly interpolated.

        Returns
        -------
        ShearLevel1
            Level 1 shear data with time, sensor speed, shear from both probes,
            and a section number array (all ones for a single cast).
        """
        microrider_data = rsIO.read_p_file(filename, self.cfg.channel_cfgs)
        time = microrider_data.header.t_fast + microrider_data.header.timestamp
        if isinstance(senspeed, tuple):
            senspeed_arr = senspeed_from_lookup_table(time, *senspeed)
        elif senspeed == "em":
            senspeed_arr = senspeed_from_em(time, microrider_data)
        else:
            senspeed_arr = senspeed_constant(time, senspeed)
        level1 = ShearLevel1(
            time=time,
            senspeed=senspeed_arr,
            cfg=cfg,
            shear=np.array(
                [
                    microrider_data.sh1.data / senspeed_arr**2,
                    microrider_data.sh2.data / senspeed_arr**2,
                ]
            ),
            section_number=np.ones_like(microrider_data.header.t_fast, dtype=int),
        )
        return level1
