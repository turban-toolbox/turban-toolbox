from abc import ABC, abstractmethod
from typing import Callable
from jaxtyping import Float

import numpy as np
from scipy.interpolate import make_interp_spline

from turban.instruments.microrider import rsIO

class SensorSpeedBase(ABC):

    def __init__(self) -> None:
        self._microrider_data: rsIO.MicroRiderData
        self._ifun: Callable

        
    @abstractmethod
    def get_sensor_speed(self, t: Float[np.ndarray, "time"]) -> Float[np.ndarray, "time"]:
        pass

    @abstractmethod
    def interpolation_factory(self) -> Callable:
        pass
    
    def set_microrider_data(self, microrider_data: rsIO.MicroRiderData) -> None:
        self.microrider_data = microrider_data
    
    
class SensorSpeedConstant(SensorSpeedBase):

    def __init__(self, constant_speed: float) -> None:
        self._constant_speed = constant_speed

    def get_sensor_speed(self, t: Float[np.ndarray, "time"]) -> Float[np.ndarray, "time"]:
        return np.ones_like(t) * self._constant_speed

    def interpolation_factory(self) -> Callable:
        raise NotImplementedError

class SensorSpeedEMC(SensorSpeedBase):

    def interpolation_factory(self) -> Callable:
        try:
            return self._ifun
        except AttributeError:
            t = self.microrider_data.header.t_slow
            t + self.microrider_data.header.timestamp
            U = self.microrider_data.U_EM.data
            self._ifun = make_interp_spline(t, U, 1)
            return self._ifun

    def get_sensor_speed(self, t: np.ndarray) -> np.ndarray:
        ifun = self.interpolation_factory()
        return ifun(t)

class SensorSpeedLookupTable(SensorSpeedBase):

    def __init__(self) -> None:
        self.data : dict[str, Float[np.ndarray, "time"]]

    def from_timeseries(self, t: Float[np.ndarray, "time"], v: Float[np.ndarray, "time"]) -> None:
        self.data["t"] = t
        self.data["U"] = U

    def interpolation_factory(self) -> Callable:
        try:
            return self._ifun
        except AttributeError:
            t = self.data["t"]
            U = self.data["U"]
            self._ifun = make_interp_spline(t, v, 1)
            return self._ifun

    def get_sensor_speed(self, t: Float[np.ndarray, "time"]) -> Float[np.ndarray, "time"]:
        ifun = self.interpolation_factory()
        return ifun(t)

