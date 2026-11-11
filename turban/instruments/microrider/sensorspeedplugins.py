from abc import ABC, abstractmethod
from collections.abc import Callable
from jaxtyping import Float

import numpy as np
from scipy.interpolate import make_interp_spline

from turban.instruments.microrider.rsCommon import MicrostructureRawData
from turban.utils.logging import get_logger

logger = get_logger(__name__)


class PluginError(Exception):
    pass


# Base class for SensorSpeed plugins


class SensorSpeedABC(ABC):

    def __init__(self) -> None:
        self._ifun: Callable[[Float[np.ndarray, "time"]], Float[np.ndarray, "time"]]
        self.microstructure_raw_data: None | MicrostructureRawData = None

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

    def set_microstructure_raw_data(
        self, microstructure_raw_data: MicrostructureRawData
    ) -> None:
        self.__dict__.pop("_ifun", None)  # Rmove _ifun if present in case of new data.
        self.microstructure_raw_data = microstructure_raw_data


# Create a registry of SensorSpeedPlugin classes:

_SENSOR_SPEED_PLUGIN_REGISTRY: dict[
    str, tuple[type[SensorSpeedABC], list[tuple[str, type]]]
] = {}


# This decorator returns a wrapper, which gets the argument of the
# class, and returns a class. To reflect this, the return type is
# defined as a callable Callable[[args], result]. See also rsCommon.py
def register_plugin(
    parameter_list: list[tuple[str, type]],
) -> Callable[[type[SensorSpeedABC]], type[SensorSpeedABC]]:
    def wrapper(cls: type[SensorSpeedABC]) -> type[SensorSpeedABC]:
        _SENSOR_SPEED_PLUGIN_REGISTRY[plugin_name(cls)] = (cls, parameter_list)
        return cls

    return wrapper


def plugin_name(cls: type[SensorSpeedABC]) -> str:
    """Return the name under which a sensor speed plugin class is registered.

    Parameters
    ----------
    cls : type[SensorSpeedABC]
        Plugin class.

    Returns
    -------
    str
        The bare class name for built-in plugins (defined in this module), or
        ``"module:ClassName"`` for custom plugins defined elsewhere.
    """
    if cls.__module__ == __name__:  # built-in plugin
        return cls.__name__
    return f"{cls.__module__}:{cls.__name__}"

def handle_custom_plugins(name: str) -> None:
    if name.count(":") == 0:
        return
    if name.count(":") == 1:
        module_name, class_name = name.split(":")
        mesg = ""
        if not module_name:
            mesg = f"Cannot import empty module. ({name})"
        elif module_name.startswith("."):
            mesg = f"Refuse to import relative modules. ({name})"
        elif not class_name:
            mesg = f"A plugin name is required. ({name})"
        else:
            import importlib
            try:
                importlib.import_module(module_name)
            except ImportError as e:
                mesg = f"Failed to import module {module_name} because {e}."
    else:
        mesg = f"Invalid plugin name ({name})."
    if mesg:
        logger.error(mesg)
        raise PluginError(mesg)
    

def _lookup_plugin(
    name: str,
) -> tuple[type[SensorSpeedABC], list[tuple[str, type]]]:
    """Return the registry entry ``(class, parameter_list)`` for `name`.

    Imports the plugin's module first if `name` is ``"module:ClassName"``.

    Parameters
    ----------
    name : str
        Registered name of the plugin.

    Returns
    -------
    tuple[type[SensorSpeedABC], list[tuple[str, type]]]
        The plugin class and its parameter list.

    Raises
    ------
    PluginError
        If `name` is invalid, its module cannot be imported, or it is not
        registered.
    """
    handle_custom_plugins(name)
    try:
        return _SENSOR_SPEED_PLUGIN_REGISTRY[name]
    except KeyError:
        mesg = (
            f"{name} is not registered as a sensor speed plugin. Registered plugins:\n"
        )
        mesg += f"    {'plugin name':20s} | parameter list\n"
        mesg += "-" * 76 + "\n"
        for k, (c, v) in _SENSOR_SPEED_PLUGIN_REGISTRY.items():
            mesg += f"    {k:20s} : {v}\n"
        logger.error(mesg)
        raise PluginError(f"{name} is not a registered SensorSpeed plugin") from None


def get_registered_plugin_parameter_list(
    name: str,
) -> list[tuple[str, type]]:
    """Return the parameter list for a registered sensor speed plugin.

    Parameters
    ----------
    name : str
        Name of the plugin class (e.g. ``"SensorSpeedConstant"``).

    Returns
    -------
    list[tuple[str, type]]
        List of ``(parameter_name, type)`` tuples describing the constructor
        arguments of the plugin. There are no defaults: every parameter must
        be set in the configuration.

    Raises
    ------
    PluginError
        If `name` is not found in the plugin registry.
    """
    _, parameter_list = _lookup_plugin(name)
    return parameter_list


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

    Raises
    ------
    PluginError
        If `name` is not a registered plugin.
    """
    C, _ = _lookup_plugin(name)
    return C(**args)


@register_plugin([("constant_speed", float)])
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
            if self.microstructure_raw_data is None:
                raise ValueError(
                    "No microstructure raw data set; call set_microstructure_raw_data() first."
                )
            t = (
                self.microstructure_raw_data.header.timestamp
                + self.microstructure_raw_data.header.t_slow
            )
            U = self.microstructure_raw_data.data["U_EM"]
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
        self.__dict__.pop("_ifun", None)  # Rmove _ifun if present in case of new data.
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


@register_plugin([("filename", str)])
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
