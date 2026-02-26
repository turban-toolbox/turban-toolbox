from turban.process.generic.config import SegmentConfig


class UTempConfig(SegmentConfig):
    """Configures processing of timeseries using temperature gradient spectra."""

    waveno_limit_upper: float = 500.0
    diff_gain: float
