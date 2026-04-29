from abc import ABC, abstractmethod
import numpy as np

from turban.instruments.generic.api import Instrument
from turban.instruments.generic.config import InstrumentConfig
import turban.instruments.microrider.sensorspeedplugins as plugins
from turban.instruments.microrider.rsCommon import ChannelConfigBaseModel
from turban.instruments.microrider import rsIO
from turban.process.shear.api import ShearLevel1
from turban.process.shear.config import ShearConfig

from turban.utils.logging import get_logger

logger = get_logger(__name__)


class MicroriderAPIError(Exception):
    pass


class MicroriderConfig(InstrumentConfig):
    sensor_speed_plugin: str = ""
    sensor_speed_plugin_parameters: dict[str, float | str] = {}
    channel_cfgs: list[ChannelConfigBaseModel] = []


class MicroriderConfig(InstrumentConfig):
    sensor_speed_plugin: str = ""
    sensor_speed_plugin_parameters: dict[str, float | str] = {}
    channel_cfgs: list[ChannelConfigBaseModel] = []

class MicroriderSonde(Instrument):

    def __init__(self, cfg: MicroriderConfig) -> None:
        self.cfg: MicroriderConfig = cfg
        self.sensor_speed_plugin: plugins.SensorSpeedABC
        self._set_sensor_speed_plugin_from_cfg()

    def set_sensor_speed_plugin(
        self, sensor_speed_plugin: plugins.SensorSpeedABC
    ) -> None:
        """Set the sensor speed plugin, overwriting any previously set plugin.

        Parameters
        ----------
        sensor_speed_plugin : plugins.SensorSpeedABC
            Plugin instance to use for computing sensor speed.
        """
        if hasattr(self, "sensor_speed_plugin"):
            new_name = sensor_speed_plugin.__class__.__name__
            old_name = self.sensor_speed_plugin.__class__.__name__
            logger.warning(
                f"Overwriting sensor speed plugin: {old_name} -> {new_name}."
            )
        self.sensor_speed_plugin = sensor_speed_plugin

    def to_shear_level1(self, filename: str, cfg: ShearConfig) -> ShearLevel1:
        """Read a MicroRider .p file and convert it to a ShearLevel1 dataclass.

        Parameters
        ----------
        filename : str
            Path to the MicroRider binary .p file.
        cfg : ShearConfig
            Processing configuration for the shear pipeline.

        Returns
        -------
        ShearLevel1
            Level 1 shear data with time, sensor speed, shear from both probes,
            and a section number array (all ones for a single cast).
        """
        channel_configs = self.cfg.channel_cfgs
        microrider_data = rsIO.read_p_file(filename, channel_configs)
        self.sensor_speed_plugin.set_microrider_data(microrider_data)
        time = microrider_data.header.t_fast + microrider_data.header.timestamp
        senspeed = self.sensor_speed_plugin.get_sensor_speed(time)
        level1 = ShearLevel1(
            time=time,
            senspeed=senspeed,
            cfg=cfg,
            shear=np.array(
                [
                    microrider_data.sh1.data / senspeed**2,
                    microrider_data.sh2.data / senspeed**2,
                ]
            ),
            section_number=np.ones_like(microrider_data.header.t_fast, dtype=int),
        )
        return level1

    def _check_for_presence_of_required_parameters(
        self, requested_plugin: str
    ) -> tuple[bool, dict[str, float | str]]:
        plugin_required_argument_list = plugins.get_registered_plugin_parameter_list(
            requested_plugin
        )
        plugin_arguments = self.cfg.sensor_speed_plugin_parameters
        # check if required arguments are given.
        argument_list_check = True
        args: dict[str, float | str] = {}
        for n, dtype, v in plugin_required_argument_list:
            if not n in plugin_arguments:
                argument_list_check = False
                break
            else:
                args[n] = v
        return argument_list_check, args

    def _check_for_unused_parameters(self, args: dict[str, str | float]) -> list[str]:
        configured_args = self.cfg.sensor_speed_plugin_parameters
        unused_args = []
        for k, v in configured_args.items():
            if k not in args:
                unused_args.append(f"{k}={v}")
        return unused_args

    def _set_sensor_speed_plugin_from_cfg(self) -> None:
        requested_plugin = self.cfg.sensor_speed_plugin
        if requested_plugin:
            argument_list_check, args = self._check_for_presence_of_required_parameters(
                requested_plugin
            )
            if not argument_list_check:
                for n in args:
                    mesg = f"Required parameter {n} for {requested_plugin} is not configured."
                    logger.error(mesg)
                raise MicroriderAPIError("Configuration error. (See logs).")
            unused_args = self._check_for_unused_parameters(args)
            if unused_args:
                mesg = f"UNUSED SensorSpeed plugin arguments: {', '.join(unused_args)}."
                logger.warning(mesg)
            # Now we need to construct the requested plugin with args
            plugin = plugins.plugin_factory(requested_plugin, args)
            self.set_sensor_speed_plugin(plugin)
