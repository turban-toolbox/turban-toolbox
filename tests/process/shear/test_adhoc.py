from turban.process.shear.api import ShearProcessing

def test_adhoc():
    p = ShearProcessing.from_atomix_netcdf(
        "data/process/shear/VMP250_TidalChannel_024.nc", level=1
    )
    print(p.level4)