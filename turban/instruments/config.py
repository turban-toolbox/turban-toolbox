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
    sampling_freq: float
    sensors: dict[str, Sensor]


class DropsondeConfig(InstrumentConfig):
    pass
