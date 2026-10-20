from dataclasses import dataclass
from numpy.typing import NDArray
from numpy import float64

@dataclass(kw_only=True)
class ByteHeader:
    file_number: int
    record_number: int
    record_number_serial_port: int
    year: int
    month: int
    day: int
    hour: int
    minute: int
    second: int
    millisecond: int
    header_version: float
    setupfile_size: int
    product_ID: int
    build_number: int
    timezone_in_minutes: int
    buffer_status: int
    restarted: int
    record_header_size: int
    data_record_size: int
    number_of_records_written: int
    frequency_clock: float
    fast_cols: int
    slow_cols: int
    n_rows: int
    data_size: int


@dataclass(kw_only=True)
class Header:
    full_path: str
    n_cols: int
    n_records: int
    fs_fast: float
    fs_slow: float
    header_version: float
    matrix_count: int
    t_slow: NDArray[float64]
    t_fast: NDArray[float64]
    timestamp: float
    date: str
    time: str

    
