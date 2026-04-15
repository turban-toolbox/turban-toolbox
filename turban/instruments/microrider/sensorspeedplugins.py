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
        self._ifun: Callable[[Float[np.ndarray, "time"]], Float[np.ndarray, "time"]]

    @abstractmethod
    def get_sensor_speed(
        self, t: Float[np.ndarray, "time"]
    ) -> Float[np.ndarray, "time"]:
        """Return sensor speed at the requested times.

        Parameters
        ----------
        t : Float[np.ndarray, "time"]
            Time vector at which sensor speed is evaluated.

        Returns
        -------
        Float[np.ndarray, "time"]
            Sensor speed in m/s, one value per element of `t`.
        """
        pass

    @abstractmethod
    def interpolation_factory(
        self,
    ) -> Callable[[Float[np.ndarray, "time"]], Float[np.ndarray, "time"]]:
        """Build and return the interpolation function for sensor speed.

        Returns
        -------
        Callable
            A callable that accepts a time vector and returns sensor speed values.
        """
        pass

    def set_microrider_data(self, microrider_data: rsIO.MicroRiderData) -> None:
        """Attach a MicroRiderData object for use by the plugin.

        Parameters
        ----------
        microrider_data : rsIO.MicroRiderData
            Data object returned by `rsIO.read_p_file`, providing access to
            channel data and header information.
        """
        self.microrider_data = microrider_data


# Create a registry of SensorSpeedPlugin classes:

_SENSOR_SPEED_PLUGIN_REGISTRY: dict[
    str, tuple[type[SensorSpeedABC], list[tuple[str, type, float | str]]]
] = {}



# This decorator returns a wrapper, which gets the argument of the
# class, and returns a class. To reflect this, the return type is
# defined as a callable Callable[[args], result]. See also rsCommon.py
def register_plugin(
    parameter_list: list[tuple[str, type, float | str]],
) -> Callable[[type[SensorSpeedABC]], type[SensorSpeedABC]]:
    def wrapper(cls: type[SensorSpeedABC]) -> type[SensorSpeedABC]:
        name = cls.__name__
        _SENSOR_SPEED_PLUGIN_REGISTRY[name] = (cls, parameter_list)
        return cls

    return wrapper


def get_registered_plugin_parameter_list(
    name: str,
) -> list[tuple[str, type, float | str]]:
    """Return the parameter list for a registered sensor speed plugin.

    Parameters
    ----------
    name : str
        Name of the plugin class (e.g. ``"SensorSpeedConstant"``).

    Returns
    -------
    list[tuple[str, type, float | str]]
        List of ``(parameter_name, type, default_value)`` tuples describing
        the constructor arguments required by the plugin.

    Raises
    ------
    PluginError
        If `name` is not found in the plugin registry.
    """
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
    """Instantiate a registered sensor speed plugin by name.

    Parameters
    ----------
    name : str
        Name of the plugin class (e.g. ``"SensorSpeedConstant"``).
    args : dict[str, float | str]
        Keyword arguments passed to the plugin constructor.

    Returns
    -------
    SensorSpeedABC
        An instance of the requested plugin.
    """
    C, _ = _SENSOR_SPEED_PLUGIN_REGISTRY[name]
    return C(**args)


@register_plugin([("constant_speed", float, 1.0)])
class SensorSpeedConstant(SensorSpeedABC):

    def __init__(self, constant_speed: float) -> None:
        """
        Parameters
        ----------
        constant_speed : float
            Sensor speed in m/s, applied uniformly for all times.
        """
        self._constant_speed = constant_speed

    def get_sensor_speed(
        self, t: Float[np.ndarray, "time"]
    ) -> Float[np.ndarray, "time"]:
        """Return a constant sensor speed for all elements of `t`.

        Parameters
        ----------
        t : Float[np.ndarray, "time"]
            Time vector; used only to determine the output shape.

        Returns
        -------
        Float[np.ndarray, "time"]
            Array of constant sensor speed values, same length as `t`.
        """
        return np.ones_like(t) * self._constant_speed

    def interpolation_factory(
        self,
    ) -> Callable[[Float[np.ndarray, "time"]], Float[np.ndarray, "time"]]:
        """Not implemented for constant sensor speed.

        Raises
        ------
        NotImplementedError
            Always, because no interpolation is needed for a constant speed.
        """
        raise NotImplementedError


@register_plugin([])
class SensorSpeedEMC(SensorSpeedABC):

    def interpolation_factory(
        self,
    ) -> Callable[[Float[np.ndarray, "time"]], Float[np.ndarray, "time"]]:
        """Build a linear spline interpolator from the EM current channel.

        The interpolator is constructed once from the slow-rate ``U_EM`` channel
        of the attached `MicroRiderData` and cached for subsequent calls.

        Returns
        -------
        Callable
            A linear spline interpolant mapping time to sensor speed.
        """
        try:
            return self._ifun
        except AttributeError:
            t = self.microrider_data.header.t_slow
            t += self.microrider_data.header.timestamp
            U = self.microrider_data.U_EM.data
            self._ifun = make_interp_spline(t, U, 1)
            return self._ifun

    def get_sensor_speed(
        self, t: Float[np.ndarray, "time"]
    ) -> Float[np.ndarray, "time"]:
        """Return sensor speed interpolated from the EM current channel.

        Parameters
        ----------
        t : Float[np.ndarray, "time"]
            Time vector at which sensor speed is evaluated.

        Returns
        -------
        Float[np.ndarray, "time"]
            Sensor speed in m/s, interpolated at each element of `t`.
        """
        ifun = self.interpolation_factory()
        return ifun(t)


@register_plugin([])
class SensorSpeedLookupTable(SensorSpeedABC):

    def __init__(self) -> None:
        self.data: dict[str, Float[np.ndarray, "time"]] = {}

    def from_timeseries(
        self, t: Float[np.ndarray, "time"], U: Float[np.ndarray, "time"]
    ) -> None:
        """Populate the lookup table from a time and speed array.

        Parameters
        ----------
        t : Float[np.ndarray, "time"]
            Time vector in seconds.
        U : Float[np.ndarray, "time"]
            Sensor speed in m/s, one value per element of `t`.
        """
        self.data["t"] = t
        self.data["U"] = U

    def interpolation_factory(
        self,
    ) -> Callable[[Float[np.ndarray, "time"]], Float[np.ndarray, "time"]]:
        """Build a linear spline interpolator from the lookup table.

        The interpolator is constructed once from the stored time and speed
        arrays and cached for subsequent calls.

        Returns
        -------
        Callable
            A linear spline interpolant mapping time to sensor speed.
        """
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
        """Return sensor speed interpolated from the lookup table.

        Parameters
        ----------
        t : Float[np.ndarray, "time"]
            Time vector at which sensor speed is evaluated.

        Returns
        -------
        Float[np.ndarray, "time"]
            Sensor speed in m/s, interpolated at each element of `t`.
        """
        ifun = self.interpolation_factory()
        return ifun(t)


@register_plugin([("filename", str, "datafile.txt")])
class SensorSpeedDataFile(SensorSpeedLookupTable):

    def __init__(self, filename: str) -> None:
        """
        Parameters
        ----------
        filename : str
            Path to a two-column text file with time in the first column
            and sensor speed in m/s in the second column.
        """
        super().__init__()
        self.filename: str = filename
        self.read_datafile()

    def read_datafile(self) -> None:
        """Load time and sensor speed from the configured text file.

        Reads `self.filename` using `numpy.loadtxt`, expecting two columns
        (time, speed), and populates the lookup table via `from_timeseries`.
        """
        t, U = np.loadtxt(self.filename).T
        self.from_timeseries(t, U)
