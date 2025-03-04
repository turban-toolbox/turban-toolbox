from turban.config import ShearConfig


def test_config():
    cfg = ShearConfig.from_atomix_netcdf("MSS_BalticSea/MSS_Baltic.nc")
