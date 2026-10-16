import numpy as np
from pydantic import BaseModel

DefaultStr = "undefined"

RawArray = np.typing.NDArray[np.int16] | np.typing.NDArray[np.float64]


class ChannelConfigBaseModel(BaseModel):
    """Base class for channel configurations

    Instrument-specific subclasses add their calibration parameters and
    override ``convert``.
    """

    name: str = ""
    id: int = 0
    type: str = "unknown"
    sign: str = "signed"
    sample_rate: float = 0.0
    units: str = ""

    def convert(self, v: RawArray) -> np.typing.NDArray[np.float64]:
        """Converts raw data into physical units. Default: no conversion."""
        return v.astype(np.float64)
