from abc import ABC, abstractmethod
from typing import Callable, TypeVar
from jaxtyping import Float

import numpy as np
from scipy.interpolate import make_interp_spline

from turban.instruments.microrider import rsIO
from turban.utils.logging import get_logger

logger = get_logger(__name__)


class PluginError(Exception):
    pass


# Base class for SensorSpeed plugins


class SensorSpeedABC(ABC):

    def __init__(self) -> None:
        self._microrider_data: rsIO.MicroRiderData
        self._ifun: Callable

    @abstractmethod
    def get_sensor_speed(
        self, t: Float[np.ndarray, "time"]
    ) -> Float[np.ndarray, "time"]:
        pass

    @abstractmethod
    def interpolation_factory(self) -> Callable:
        pass

    def set_microrider_data(self, microrider_data: rsIO.MicroRiderData) -> None:
        self.microrider_data = microrider_data


# Create a registry of SensorSpeedPlugin classes:

T = TypeVar("T", bound=SensorSpeedABC)  # A type machting also all its derivates.
_SENSOR_SPEED_PLUGIN_REGISTRY: dict[str, tuple[type[T], type[SensorSpeedABC]]] = {}


# This decorator returns a wrapper, which gets the argument of the
# class, and returns a class. To reflect this, the return type is
# defined as a callable Callable[[args], result]. See also rsCommon.py
def register_plugin(
    parameter_list: list[tuple[str, type, float | str]],
) -> Callable[type[T], type[T]]:
    def wrapper(cls: type[T]) -> type[T]:
        name = cls.__name__
        _SENSOR_SPEED_PLUGIN_REGISTRY[name] = (cls, parameter_list)
        return cls

    return wrapper


def get_registered_plugin_parameter_list(
    name: str,
) -> list[tuple[str, type, float | str]]:
    try:
        _, parameter_list = _SENSOR_SPEED_PLUGIN_REGISTRY[name]
    except KeyError:
        mesg = (
            f"{name} is not registered as a sensor speed plugin. Registered plugins:\n"
        )
        mesg += f"    {'plugin name':20s} | parameter list\n"
        mesg += "-" * 76 + "\n"
        for k, (c, v) in _SENSOR_SPEED_PLUGIN_REGISTRY.items():
            mesg += f"    {k:20s} : {v}\n"
        logger.error(mesg)
    else:
        return parameter_list
    # If we get here, we didn't manage to return the parameter list.
    raise PluginError(f"{name} is not a registered SensorSpeed plugin")


def plugin_factory(name: str, args: dict[str, float | str]) -> SensorSpeedABC:
    C, _ = _SENSOR_SPEED_PLUGIN_REGISTRY[name]
    c = C(**args)
    return c


@register_plugin([("constant_speed", float, 1.0)])
class SensorSpeedConstant(SensorSpeedABC):

    def __init__(self, constant_speed: float) -> None:
        self._constant_speed = constant_speed

    def get_sensor_speed(
        self, t: Float[np.ndarray, "time"]
    ) -> Float[np.ndarray, "time"]:
        return np.ones_like(t) * self._constant_speed

    def interpolation_factory(self) -> Callable:
        raise NotImplementedError


class SensorSpeedEMC(SensorSpeedABC):

    def interpolation_factory(self) -> Callable:
        try:
            return self._ifun
        except AttributeError:
            t = self.microrider_data.header.t_slow
            t + self.microrider_data.header.timestamp
            U = self.microrider_data.U_EM.data
            self._ifun = make_interp_spline(t, U, 1)
            return self._ifun

    def get_sensor_speed(
        self, t: Float[np.ndarray, "time"]
    ) -> Float[np.ndarray, "time"]:
        ifun = self.interpolation_factory()
        return ifun(t)


class SensorSpeedLookupTable(SensorSpeedABC):

    def __init__(self) -> None:
        self.data: dict[str, Float[np.ndarray, "time"]]

    def from_timeseries(
        self, t: Float[np.ndarray, "time"], U: Float[np.ndarray, "time"]
    ) -> None:
        self.data["t"] = t
        self.data["U"] = U

    def interpolation_factory(self) -> Callable:
        try:
            return self._ifun
        except AttributeError:
            t = self.data["t"]
            U = self.data["U"]
            self._ifun = make_interp_spline(t, U, 1)
            return self._ifun

    def get_sensor_speed(
        self, t: Float[np.ndarray, "time"]
    ) -> Float[np.ndarray, "time"]:
        ifun = self.interpolation_factory()
        return ifun(t)
