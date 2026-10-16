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

    def __repr__(self) -> str:
        s: str = f"{self.__class__.__name__}:\n"
        for k in type(self).model_fields:
            if k in self.model_fields_set:
                s += f"    {k}: {getattr(self, k)}\n"
        return s

    def convert(self, v: RawArray) -> np.typing.NDArray[np.float64]:
        """Converts raw data into physical units. Default: no conversion."""
        return v.astype(np.float64)
