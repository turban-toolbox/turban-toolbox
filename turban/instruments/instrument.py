from typing import List, Dict, Optional
from pydantic import BaseModel


class Sensor(BaseModel):
    name: str
    coefficients: List[float]
    channel: int
    calibration_type: str


class ShearSensor(Sensor):
    sensitivity: float
    serial_number: str
    reference_temperature: float
    calibration_date: str


class Instrument(BaseModel):
    sampling_freq: float
    sensors: Dict[str, Sensor]


class Dropsonde(Instrument):
    pass
