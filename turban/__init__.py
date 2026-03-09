from beartype.claw import beartype_this_package
from importlib.metadata import version as _pkg_version, PackageNotFoundError

beartype_this_package()

try:
    __version__ = _pkg_version("turban-toolbox")
except PackageNotFoundError:
    __version__ = "0+unknown"

from turban.process.shear.api import (
    ShearProcessing,
    ShearConfig,
    ShearLevel1,
    ShearLevel2,
    ShearLevel3,
    ShearLevel4,
)
from turban.utils.plot.shear import plot
