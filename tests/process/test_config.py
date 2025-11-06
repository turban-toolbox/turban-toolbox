from turban.process.generic.config import SegmentConfig
from turban.process.shear.config import ShearConfig
from tests.fixtures import atomix_mss_nc_filename


def test_config(atomix_mss_nc_filename):
    cfg = SegmentConfig.from_atomix_netcdf(atomix_mss_nc_filename)
    cfg = ShearConfig.from_atomix_netcdf(atomix_mss_nc_filename)
