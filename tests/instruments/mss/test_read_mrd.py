from tests.fixtures import mss_mrd_filename
import turban

def test_read_mrd(mss_mrd_filename):
    mrddata = turban.instruments.mss.mss_mrd.mrd(mss_mrd_filename)
    print(mrddata.level0['PRESS'])
