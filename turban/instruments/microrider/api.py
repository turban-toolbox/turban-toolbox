from abc import ABC, abstractmethod

import numpy as np

from turban.instruments.generic.api import Instrument
from turban.instruments.generic.config import InstrumentConfig
import turban.instruments.microrider.sensorspeedplugins as plugins
from turban.instruments.microrider import rsIO
from turban.process.shear.api import ShearLevel1
from turban.process.shear.config import ShearConfig



class MicroRiderBaseSonde(Instrument):
    def __init__(self, cfg: InstrumentConfig) -> None:
        self.cfg = cfg
        self.sensor_speed_plugin: plugins.SensorSpeedBase
    
    @abstractmethod
    def set_sensor_speed_plugin(self, sensor_speed_plugin: plugins.SensorSpeedBase) -> None:
        pass
    
class MicroriderSlocumSonde(MicroRiderBaseSonde):

    def __init__(self, cfg: InstrumentConfig) -> None:
        self.cfg = cfg
        

    def set_sensor_speed_plugin(self, sensor_speed_plugin: plugins.SensorSpeedBase) -> None:
        self.sensor_speed_plugin = sensor_speed_plugin

    def to_shear_level1(self, p_filename : str, cfg:ShearConfig) -> ShearLevel1:
        microrider_data = rsIO.read_p_file(p_filename)
        self.sensor_speed_plugin.set_microrider_data(microrider_data)
        time = microrider_data.header.t_fast + microrider_data.header.timestamp
        senspeed = self.sensor_speed_plugin.get_sensor_speed(time)
        level1 = ShearLevel1(
            time=time,
            senspeed=senspeed,
            cfg = cfg,
            shear=np.array([microrider_data.sh1.data/senspeed**2,microrider_data.sh2.data/senspeed**2]),
            section_number=np.ones_like(microrider_data.header.t_fast, dtype=int)
        )
        return level1
        
