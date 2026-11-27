from pydantic import BaseModel


class InstrumentConfig(BaseModel):
    """Base class of instrument configurations; instruments add their own fields."""


class DropsondeConfig(InstrumentConfig):
    pass


class ThermistorElectronics(BaseModel):
    """Calibration and electronics of one FP07 thermistor channel (pre-emphasised)."""

    diff_gain: float  # [s] pre-emphasis gain
