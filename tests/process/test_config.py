from turban.process.generic.config import SegmentConfig
from turban.process.shear.config import ShearConfig
from tests.filepaths import atomix_benchmark_baltic_fpath, atomix_benchmark_faroe_fpath


def test_config():
    for path in [atomix_benchmark_baltic_fpath, atomix_benchmark_faroe_fpath]:
        for Config in [SegmentConfig, ShearConfig]:
            Config.from_atomix_netcdf(path)
