from turban.process.generic.config import SegmentConfig


class TempConfig(SegmentConfig):
    """Configures processing of timeseries using temperature gradient spectra."""
    waveno_limit_upper: float = 500.0
