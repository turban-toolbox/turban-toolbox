from pathlib import Path
from pytest import fixture

top_level = Path(__file__).resolve().parent.parent

@fixture
def atomix_nc_filename():
    return str(top_level / "data" / "mss" / "MSS_Baltic.nc")

@fixture
def mss_mrd_filename():
    return str(top_level / "data" / "mss" / "Nien0020.MRD")

