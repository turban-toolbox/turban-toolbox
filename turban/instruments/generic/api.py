from abc import abstractmethod, ABC
from turban.instruments.generic.config import InstrumentConfig
from turban.process.shear.api import ShearLevel1
from turban.process.shear.config import ShearConfig


class Instrument(ABC):
    def __init__(self, cfg: InstrumentConfig):
        self.cfg = cfg
        
    @abstractmethod
    def to_shear_level1(self, filename: str, cfg: ShearConfig) -> "ShearLevel1":
        """
        Convert raw data to shear level 1
        """
        pass
    

class Dropsonde(Instrument):

    def to_shear_level1(self) -> "ShearLevel1":
        """
        Convert raw data to shear level 1
        """
        raise NotImplementedError("to_shear_level1 must be implemented in subclasses")
