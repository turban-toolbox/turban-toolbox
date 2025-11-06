from pathlib import Path
from pytest import fixture

top_level = Path(__file__).resolve().parent.parent


@fixture
def atomix_mss_nc_filename():
    return str(top_level / "data" / "mss" / "MSS_Baltic.nc")

@fixture
def atomix_mss_mrd_filename():
    return str(top_level / "data" / "mss" / "SH2_0330.MRD")


@fixture
def mss_mrd_filename():
    return str(top_level / "data" / "mss" / "Nien0020.MRD")
