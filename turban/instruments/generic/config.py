from pydantic import BaseModel


class Sensor(BaseModel):
    name: str
    coefficients: list[float]
    channel: int
    calibration_type: str


class ShearSensor(Sensor):
    sensitivity: float
    serial_number: str
    reference_temperature: float
    calibration_date: str


class InstrumentConfig(BaseModel):
    sampfreq: float
    sensors: dict[str, Sensor]


class DropsondeConfig(InstrumentConfig):
    pass

class ThermistorElectronics(BaseModel):
    """Calibration and electronics of one FP07 thermistor channel (pre-emphasised).
    """

    diff_gain: float  # [s] pre-emphasis gain