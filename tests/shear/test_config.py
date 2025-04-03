from turban.shear.config import ShearConfig
from tests.fixtures import atomix_nc_filename


def test_config(atomix_nc_filename):
    cfg = ShearConfig.from_atomix_netcdf(atomix_nc_filename)
