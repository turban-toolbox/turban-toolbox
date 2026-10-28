import enum

from functools import partial
import io
import os.path
import re
import struct
from typing import Any, Self

import arrow
import numpy as np
from scipy.interpolate import CubicSpline as si_CubicSpline

from turban.instruments.generic.api import InstrumentEnum
from turban.instruments.generic.channels import (
    ChannelConfigBaseModel,
    channel_config_factory,
)
from turban.instruments.microrider import rsConversions
from turban.instruments.microrider import rsConfig_parser
from turban.instruments.microrider.rsCommon import (
    ByteHeader,
    HeaderInfo,
)

from turban import logger_manager

logger = logger_manager.get_logger(__name__)


class HeaderEnum(enum.IntEnum):
    HeaderSize = 128
    WordSize = 2
    HeaderFields = HeaderSize // WordSize
    Endianness = 63


class uriderException(Exception):
    pass


class Channel(object):
    """Channel object

    Parameters
    ----------
    config : ChannelConfig
        channel configuration object (from setupstr)

    The following methods are provided:

    copy(): this returns a new channel, but keeps the same configuration,
            and sets the deconvolved flag. This method is used to create
            new channels for the high resolution parameters

    correct_sign(): sign correction (u2 -> i2) if configuration dictates.

    convert_to_units(): applies to associated conversion algortithm (determined
           from the config dictionary) to the raw data.

    """

    def __init__(
        self, channel_config: ChannelConfigBaseModel) -> None:
        self.name: str = channel_config.name
        self.config: ChannelConfigBaseModel = channel_config
        self.converter = rsConversions.get_converter(channel_config)(channel_config)
        self.converted = False

    def convert_to_units(self,
                         records: np.typing.NDArray[np.int16] | np.typing.NDArray[np.float64],
                         skip_decode=False
                         ) -> np.typing.NDArray[np.float64]:
        """Converts channel into unit bearing data

        Sets the converted_into_units flag.
        """
        if not self.converted:
            self.converted = True
            if skip_decode: # deconvolution channels have the data
                            # already decoded and will set this flag
                            # to True
                data = self.converter.convert(records)
            else:
                data = self.converter(records)
        else:
            raise ValueError(
                "Refusing to convert to units for a parameter already converted."
            )
        return data

    def deconvolve(self,
                   records: np.typing.NDArray[np.int16],
                   co_channel: np.typing.NDArray[np.float64]) -> np.typing.NDArray[np.float64]:
        pass

    def reset(self) -> None:
        self.converted = False

class ChannelMatrix(object):
    """Class to handle the sensor matrix

    Parameters
    ----------
    microrider_config : dict[str, str|int|float|list[int]]
        dictionary holding the MR configuration

    """

    def __init__(self,
                 microrider_config: rsConfig_parser.MicroRiderConfig,
                 header_data: ByteHeader,
                 header_info: HeaderInfo) -> None:
        self.matrix = self._get_matrix(microrider_config)
        self.header_info = header_info # header information to pass
        self.header_data = header_data # header information extracted from file.
        self.channels = self._create_channels(microrider_config)
        self._set_sample_rates()
        self._regex_deconvolve_preemph_name = re.compile(r"([TP].?)_d([TP].?)")
        self._regex_deconvolve_name = re.compile(r"^(T[12]|P)$")
        
        
    def get_record_indices(self, channel_name: str) -> np.typing.NDArray[np.int64]:
        n_matrix_rows, _ = self.matrix.shape
        channel_positions = self.matrix.flatten()
        channel = self.channels[channel_name]
        indices = np.where(channel_positions == channel.config.id)[0]
        return indices

    def bytes_to_records(self, data_bytes: bytes) -> np.typing.NDArray[np.int16]:
        endianness = self.header_data.endianness
        dtype_data = np.dtype(f"{endianness}i{HeaderEnum.WordSize}")
        bindata = np.frombuffer(data_bytes, dtype=dtype_data).reshape(
            -1, self.header_data.data_record_size
        )
        # Make sure we end up with int16, as we may get '>i2'
        matrix_size = np.prod(self.matrix.shape)
        data = bindata[:, self.header_data.record_header_size :].reshape(
            -1, matrix_size).astype(np.int16)
        return data
        
    def convert_channels(self, data_bytes: bytes) -> dict[str,np.typing.NDArray[np.float64]]:
        """
        """
        data = self.bytes_to_records(data_bytes)
        data_dict: dict[str, np.typing.NDArray[np.float64]] = {}

        # First any channels that require deconvolution
        for name, channel in self.channels.items():
            regex_preemph = self._regex_deconvolve_preemph_name.match(name)
            if regex_preemph: # preemphasis channel
                logger.info(f"Converting via convolution {name}")
                indices = self.get_record_indices(name)
                records = data[:, indices].ravel()
                # Decode both channels at their native sample rates. As in
                # ODAS, the deconvolution runs at the rate of the
                # preemphasis channel; a slower co-channel is interpolated
                # inside Deconvolve.
                X_dX = channel.converter.decode(records)
                # get co-channel data
                co_channel_name = self.get_co_channel_name(name)
                co_channel = self.channels[co_channel_name]
                indices = self.get_record_indices(co_channel_name)
                records_co_channel = data[:, indices].ravel()
                X = co_channel.converter.decode(records_co_channel)
                if hasattr(channel.config, "diff_gain"):
                    diff_gain = channel.config.diff_gain  # <- is required to be present
                else:
                    raise AttributeError(
                        "diff_gain is not an attribute for this configuration object."
                    )
                # The deconvolved data are written to the
                # co_channel_name. The preemphasis data are dropped.
                X_hires = rsConversions.Deconvolve(
                    X_dX, X, channel.config.sample_rate, diff_gain
                ).X_hires
                # Set skip_decode flat to True because we don't want implicit decoding.
                X_hires = co_channel.convert_to_units(X_hires, skip_decode=True)
                # Unit-bearing data are put onto the fast time base, as in ODAS.
                data_dict[co_channel_name] = self._interpolate_to_fast(X_hires)
        
        # Then do all normal channels (no deconvolution step required).
        for name, channel in self.channels.items():
            regex = self._regex_deconvolve_name.match(name)
            regex_preemph = self._regex_deconvolve_preemph_name.match(name)
            if not regex and not regex_preemph: # normal channel
                indices = self.get_record_indices(name)
                records = data[:, indices].ravel()
                logger.info(f"Converting {name}")
                data_dict[name] = channel.convert_to_units(records)

        # Reset the converted flag for re-use of channels.
        for channel in self.channels.values():
            channel.reset()
        return data_dict



    def get_co_channel_name(self, name: str) -> str:
        # Note: we rely on whether there is a co channel or not has been decided upon before.
        co_channel_name, _ = name.split("_")
        return co_channel_name

    def _interpolate_to_fast(self, v: np.typing.NDArray[np.float64]) -> np.typing.NDArray[np.float64]:
        """Interpolates v onto the fast time base (ODAS interp1_if_req: spline, extrapolated)"""
        t_fast = self.header_info.t_fast
        if v.shape[0] == t_fast.shape[0]:
            return v
        t = np.arange(v.shape[0]) / (self.header_info.fs_fast * v.shape[0] / t_fast.shape[0])
        ifun = si_CubicSpline(t, v, bc_type="not-a-knot", extrapolate=True)
        return np.asarray(ifun(t_fast), dtype=np.float64)
    
    def _set_sample_rates(self) -> None:
        n_rows = self.matrix.shape[0]
        for name, channel in self.channels.items():
            n_samples_per_matrix = len(self.get_record_indices(name))
            channel.config.update("sample_rate", self.header_info.fs_fast * n_samples_per_matrix / n_rows)
            
    def _get_matrix(
        self, microrider_config: rsConfig_parser.MicroRiderConfig
    ) -> np.typing.NDArray[np.int64]:
        matrix = []
        matrix_section = microrider_config.get_section("matrix")
        # we don't always have the number of rows given anymore.
        # Find out ourselves
        n = 0
        while True:
            k = f"row{n+1:02d}"  # Here they count from 1, for variation I suppose.
            if not k in matrix_section:
                break
            s = matrix_section[k]
            matrix.append(s)
            n += 1
        logger.info("Address matrix:")
        for row in matrix:
            _s = " ".join([f"{i:3d}" for i in list(row)])
            logger.info(f"| {_s} |")
        return np.array(matrix)

    def _create_channels(
        self, microrider_config: rsConfig_parser.MicroRiderConfig
    ) -> dict[str, Channel]:
        """ """
        channels: dict[str, Channel] = {}

        logger.info("Available channels:")
        for n in range(microrider_config.number_of_channels):
            section_name = f"channel{n:02d}"
            section_dict = microrider_config.get_section(section_name)
            channel_config = self._create_channel_config(section_dict)
            channel_name = channel_config.name
            channels[channel_name] = Channel(channel_config)
            logger.info(f"\t{n:2d}: {channel_name}")
        if np.any(self.matrix == 255):
            logger.debug("Created Channel 255...")
            channel_config = channel_config_factory("ch255", prefix=InstrumentEnum.MicroRider)
            channel_config.id = 255
            channel_config.type = ""
            channels["ch255"] = Channel(channel_config)
            n += 1
            logger.info(f"\t{n:2d}: ch255")
        return channels 

    def _create_channel_config(self, section: dict[str, Any]) -> ChannelConfigBaseModel:
        name = section["name"]
        channel_config = channel_config_factory(name, prefix=InstrumentEnum.MicroRider)
        for k_any_case, v in section.items():
            k = k_any_case.lower()
            if k == "name":
                continue
            if k == "id":
                if not isinstance(v, int):
                    raise NotImplementedError(
                        "Split channels are not implemented. Exiting."
                    )
            channel_config.update(k, v)
        return channel_config

    
class HeaderParser:
    def parse(self, fd: io.BufferedReader) -> ByteHeader:
        """Parses header block of a data record. It is assumed that the block is HeaderEnum.HeaderSize large.

        Parameters
        ----------
        fd : io.BufferedReader
            file descriptor pointing to an open file

        Returns
        -------
        ByteHeader:
            a dataclass with (byte) header information
        """
        block = fd.read(HeaderEnum.HeaderSize)
        match self.__get(HeaderEnum.Endianness, block=block, fmt="<H"):
            case 0:
                get = partial(self.__get, block=block, fmt="<H")
                endianness="<"
            case 1 | 256:  # little endian
                get = partial(self.__get, block=block, fmt="<H")
                endianness="<"
            case 2 | 512:  # big endian
                get = partial(self.__get, block=block, fmt=">H")
                endianness=">"
            case _:
                raise ValueError(
                    "Failed to determine the endianness of this block of data."
                )

        # From the manual...
        byte_header = ByteHeader(
            file_number=get(0),
            record_number=get(1),
            record_number_serial_port=get(2),
            year=get(3),
            month=get(4),
            day=get(5),
            hour=get(6),
            minute=get(7),
            second=get(8),
            millisecond=get(9),
            header_version=float(get(10) >> 8) + float((get(10) & 0x0F)) / 1000,
            setupfile_size=get(11),
            product_ID=get(12),
            build_number=get(13),
            timezone_in_minutes=get(14),
            buffer_status=get(15),
            restarted=get(16),
            record_header_size=get(17) // HeaderEnum.WordSize,
            data_record_size=get(18) // HeaderEnum.WordSize,
            number_of_records_written=get(19),
            frequency_clock=float(get(20)) + float(get(21)) / 1e3,
            fast_cols=get(28),
            slow_cols=get(29),
            n_rows=get(30),
            data_size=(get(18) - get(17)) // HeaderEnum.WordSize,
            endianness=endianness,
            filename=fd.name,
        )
        return byte_header

    def __get(self, pos: int, block: bytes = b"", fmt: str = "<H") -> int:
        p = pos * HeaderEnum.WordSize
        v = int(struct.unpack(fmt, block[p : p + HeaderEnum.WordSize])[0])
        return v

    def check_for_bad_blocks(self, fp: io.BufferedReader) -> int:
        """Checks for bad blocks

        Method to check for bad blocks. The method reads all headers,
        and requires that the number of data blocks read matches the
        number of data blocks as written in the header. If these do
        not match, this points to an error in the data file.

        If something does not match, an exception is thrown.

        Parameters
        ----------
        fp: io.BufferedReader
            file descriptor pointing to an open file

        Returns
        -------
        int:
            number of data blocks read.

        """
        current_fp_location: int = fp.tell()
        fp.seek(0, 2)  # go to the end
        file_size: int = fp.tell()
        fp.seek(0, 0)  # rewind
        record_number: int = 0
        bytes_read: int = 0
        while True:
            # we are going to read 128 bytes. Make sure that that is possible
            if bytes_read == file_size:
                break  # we are done
            if file_size - bytes_read > HeaderEnum.HeaderSize:
                byte_header = self.parse(fp)
                bytes_read += HeaderEnum.HeaderSize
                if byte_header.record_number == record_number:
                    if byte_header.record_number == 0:
                        s = byte_header.setupfile_size
                    else:
                        s = byte_header.data_size * HeaderEnum.WordSize
                    bytes_read += s
                    if bytes_read > file_size:
                        raise uriderException("File seems truncated.")
                    fp.seek(s, 1)
                    record_number += 1
                else:
                    raise uriderException("Missing record(s).")
            else:
                # we cannot read header file, although there are bytes to read.
                raise uriderException("File seems truncated.")
        fp.seek(current_fp_location, 0)  # go back to where we started.
        record_number -= (
            1  # We count the first record too, but this is not a data record.
        )
        return record_number

    def read_setupstring(self, fd: io.BufferedReader, byte_header: ByteHeader) -> str:
        """Reads the setupstring from the binary file

        Parameters
        ----------
        fd : file descriptor

        byte_header: ByteHeader
            ByteHeader dataclass

        Returns
        -------
        str
            setupstring.

        """
        fd_current = fd.tell()
        fd.seek(HeaderEnum.HeaderSize, 0)
        setupfile_size = byte_header.setupfile_size
        bytestr = fd.read(setupfile_size)
        s = bytestr.decode()
        fd.seek(fd_current, 0)
        return s


    def get_header_info(self, byte_header: ByteHeader, n_records: int) -> HeaderInfo:
        """

        Parameters
        ----------
        byte_header : ByteHeader
            ByteHeader dataclass
        """
        n_cols = byte_header.fast_cols + byte_header.slow_cols
        matrix_count = (
            n_records * byte_header.data_size // (byte_header.n_rows * n_cols)
        )
        fs_fast = byte_header.frequency_clock / n_cols
        fs_slow = byte_header.frequency_clock / n_cols / byte_header.n_rows
        header_version = byte_header.header_version
        t_slow = np.arange(matrix_count) / fs_slow
        t_fast = np.arange(matrix_count * byte_header.n_rows) / fs_fast
        timestamp, datestring, timestring = self.get_file_timestamp(byte_header)

        header_info = HeaderInfo(
            full_path=os.path.realpath(byte_header.filename),
            n_cols=n_cols,
            n_records=n_records,
            fs_fast=fs_fast,
            fs_slow=fs_slow,
            header_version=header_version,
            matrix_count=matrix_count,
            t_fast=t_fast,
            t_slow=t_slow,
            timestamp=timestamp,
            date=datestring,
            time=timestring,
        )
        return header_info
    
    def get_file_timestamp(self, byte_header: ByteHeader) -> tuple[float, str, str]:
        """Gets time stamp of the file

        Parameters
        ----------
        byte_header : ByteHeader
            ByteHeader dataclass with the header parameters

        Returns
        -------
        (float, str, str):
            timestamp, date string, time string

        The methods return time stamp (seconds since 1970), date string
        and time string

        """
        date = arrow.get(
            byte_header.year,
            byte_header.month,
            byte_header.day,
            byte_header.hour,
            byte_header.minute,
            byte_header.second,
            byte_header.millisecond,
        )
        timestamp = date.timestamp()
        datestr = date.strftime("%Y-%m-%d")
        timestr = date.strftime("%H-%M-%S")
        return timestamp, datestr, timestr


    
# class MicroRiderData(object):
#     """Data class to hold all data collected by the RSI MicroRider.

#     parameters
#     ----------

#     path : string
#         full path to .P file

#     config: system configuration, extracted from the setupfile string.

#     """

#     def __init__(self, path: str, config: rsConfig_parser.MicroRiderConfig) -> None:
#         self.config: rsConfig_parser.MicroRiderConfig = config
#         self.header: Header
#         self.channel_matrix: ChannelMatrix = ChannelMatrix(config)
#         self._regex_deconvolve_preemph_name = re.compile(r"([TP].?)_d([TP].?)")
#         self._regex_deconvolve_name = re.compile(r"^(T[12]|P)$")
#         self._path = path

#     def __getattr__(self, name: str) -> Channel:
#         try:
#             return self.channel_matrix.channels[name]
#         except KeyError:
#             raise AttributeError(
#                 f"'{self.__class__.__name__}' object has no attribute '{name}'"
#             )

#     def keys(self) -> list[str]:
#         return list(self.channel_matrix.channels.keys())

#     def add_header_data(self, byte_header: ByteHeader, n_records: int) -> None:
#         """Adds specific parameters from the byteheader to the data structure

#         Parameters
#         ----------
#         byte_header : ByteHeader
#             ByteHeader dataclass
#         """
#         n_cols = byte_header.fast_cols + byte_header.slow_cols
#         matrix_count = (
#             n_records * byte_header.data_size // (byte_header.n_rows * n_cols)
#         )
#         fs_fast = byte_header.frequency_clock / n_cols
#         fs_slow = byte_header.frequency_clock / n_cols / byte_header.n_rows
#         header_version = byte_header.header_version
#         t_slow = np.arange(matrix_count) / fs_slow
#         t_fast = np.arange(matrix_count * byte_header.n_rows) / fs_fast
#         timestamp, datestring, timestring = self.get_file_timestamp(byte_header)

#         self.header = Header(
#             full_path=os.path.realpath(self._path),
#             n_cols=n_cols,
#             n_records=n_records,
#             fs_fast=fs_fast,
#             fs_slow=fs_slow,
#             header_version=header_version,
#             matrix_count=matrix_count,
#             t_fast=t_fast,
#             t_slow=t_slow,
#             timestamp=timestamp,
#             date=datestring,
#             time=timestring,
#         )

#     def get_file_timestamp(self, byte_header: ByteHeader) -> tuple[float, str, str]:
#         """Gets time stamp of the file

#         Parameters
#         ----------
#         byte_header : ByteHeader
#             ByteHeader dataclass with the header parameters

#         Returns
#         -------
#         (float, str, str):
#             timestamp, date string, time string

#         The methods return time stamp (seconds since 1970), date string
#         and time string

#         """
#         date = arrow.get(
#             byte_header.year,
#             byte_header.month,
#             byte_header.day,
#             byte_header.hour,
#             byte_header.minute,
#             byte_header.second,
#             byte_header.millisecond,
#         )
#         timestamp = date.timestamp()
#         datestr = date.strftime("%Y-%m-%d")
#         timestr = date.strftime("%H-%M-%S")
#         return timestamp, datestr, timestr
    
#     def add_channel_data(self, data_snr: np.typing.NDArray[np.int16]) -> None:
#         """Adds the channel data, from array with dtype i2/u2

#         Parameters
#         ----------

#         data_snr : array
#             array with bytes converted to signed and unsigned 16 bit integers.

#         The method adds all the numeric data conveyed in this matrix to the specific
#         channels. It uses the sensor matrix to determine which part goes to what
#         channel.

#         Note: this channel adds a keyword to the configuration: depending on the
#               number of values per record, it adds whether this is a fast or
#               slow senor if the frequency is 512 or 64, respectively.
#         """
#         n_matrix_rows, _ = self.channel_matrix.matrix.shape
#         channel_positions = self.channel_matrix.matrix.flatten()
#         for channel_name, channel in self.channel_matrix.channels.items():
#             logger.debug(f"Channel name : {channel_name}")
#             i_channel = channel.config.id
#             idx = np.where(channel_positions == i_channel)[0]
#             n_data_per_record = len(idx)
#             if n_data_per_record == n_matrix_rows:
#                 channel.config.update("sample_rate", self.header.fs_fast)
#             elif n_data_per_record == 1:
#                 channel.config.update("sample_rate", self.header.fs_slow)
#             else:
#                 channel.config.update(
#                     "sample_rate", 0
#                 )  # we don't need to know this for this channel anyway.
#             # correct the sign of the data if necessary
#             channel.data = data_snr[:, idx].flatten()
#             channel.correct_sign()


#     def convert_channels(self) -> None:
#         """Converts all channels

#         Converts all channels to data with physical units. If a channel
#         needs to be deconvolved first, this is also done (and leads to
#         a new channel).
#         """
#         for n, ch in self.channel_matrix.channels.copy().items():
#             regex_preemph = self._regex_deconvolve_preemph_name.match(n)
#             regex = self._regex_deconvolve_name.match(n)
#             if regex_preemph:
#                 # These channels need first to be deconvolved.
#                 logger.info(f"Deconvolving and converting {n}")
#                 self.deconvolve_and_convert_channel(ch)
#             elif (
#                 not regex
#             ):  # skipping T1 T2 and P, because they are interpolated from the deconvolved versions
#                 logger.info(f"Converting {n}")
#                 ch.convert_to_units()
#             else:
#                 logger.debug(
#                     f"Skipping {n} for now, because, they are interpolated from the deconvolved versions???"
#                 )

#     def deconvolve_and_convert_channel(self, ch: Channel) -> None:
#         """Treats channels that need deconvolving separately

#         Note: this method creates a new channel, with the name
#               based on the primary channel (e.g. T1) with "_hires" added.


#         """
#         channel_name = ch.name
#         co_channel_name, _ = ch.name.split("_")
#         co_channel = self.channel_matrix.channels[co_channel_name]
#         # this channel should not be converted yet.
#         if ch.converted_into_units:
#             raise ValueError(
#                 "This preemph channel is converted to units, but should be deconvolved first."
#             )
#         if co_channel.converted_into_units:
#             raise ValueError(
#                 "The channel (without preemphasis) is converted to units, but should be usedin the deconvolution first."
#             )
#         logger.debug(f"Trying to deconvolve {channel_name} ({co_channel_name})")
#         if not np.isclose(ch.config.sample_rate, self.header.fs_fast):
#             ch_fast = self.interpolate_onto_fast_channel(ch)
#         else:
#             ch_fast = ch
#         if not np.isclose(co_channel.config.sample_rate, self.header.fs_fast):
#             co_channel_fast = self.interpolate_onto_fast_channel(co_channel)
#         else:
#             co_channel_fast = co_channel
#         X = co_channel_fast.data  # not converted yet, so OK.
#         X_dX = ch_fast.data
#         if hasattr(ch_fast.config, "diff_gain"):
#             diff_gain = ch_fast.config.diff_gain  # <- is required to be present
#         else:
#             raise AttributeError(
#                 "diff_gain is not an attribute for this configuration object."
#             )
#         fs = self.header.fs_fast

#         new_channel_name = f"{co_channel_name}_hires"
#         new_channel = co_channel.copy(
#             new_channel_name
#         )  # copy config from T1, name it T1_res
#         new_channel.data = rsConversions.Deconvolve(
#             X_dX.astype(np.float64), X.astype(np.float64), fs, diff_gain
#         ).X_hires
#         new_channel.convert_to_units()
#         self.channel_matrix.channels[new_channel_name] = new_channel
#         # interpolate co channel.
#         logger.info(
#             f"Converting {co_channel_name} by means of interpolation of unit-bearing hires channel"
#         )
#         self.interpolate_co_channel(co_channel, new_channel)

#     def interpolate_onto_fast_channel(self, ch: Channel) -> Channel:
#         logger.debug(f"Interpolating {ch.name} to a fast channel...")
#         ch_hires = ch.copy("dummy")
#         ch_hires.config.sample_rate = self.header.fs_fast
#         ifun = si_interp1d(
#             self.header.t_slow, ch.data, kind="cubic", fill_value="extrapolate"
#         )
#         ch_hires.data = ifun(self.header.t_fast).astype(np.float64)
#         return ch_hires

#     def interpolate_co_channel(
#         self, co_channel: Channel, hires_channel: Channel
#     ) -> None:
#         if np.isclose(co_channel.config.sample_rate, self.header.fs_slow):
#             tm_i = self.header.t_slow
#             ifun = si_interp1d(
#                 self.header.t_fast,
#                 hires_channel.data,
#                 kind="cubic",
#                 fill_value="extrapolate",
#             )
#             co_channel.data = ifun(tm_i).astype(np.float64)
#             co_channel.converted_into_units = True
#         else:
#             raise ValueError(
#                 "Trying to interpolate to hires from a hires vector. Fix me."
#             )


        
def read_p_file(filename: str,
                channel_configs: list[ChannelConfigBaseModel]=[]) -> dict[str,np.typing.NDArray[np.float64]]:
    """Function to read a single .p file.

    Parameters
    ----------
    filename : str
         Name of .p file to read

    Returns
    -------
    MicroRiderData object
        Raw data, converted into physical units.
    """
    header_parser = HeaderParser()
    microrider_config = rsConfig_parser.MicroRiderConfig()
    full_path = os.path.realpath(filename)

    with open(filename, "rb") as fd:
        header_parser = HeaderParser()
        header_data = header_parser.parse(fd)
        n_records = header_parser.check_for_bad_blocks(fd)
        header_info = header_parser.get_header_info(header_data, n_records)
        # read embedded setupstring (leaves the fp in the correct place)
        setupstring = header_parser.read_setupstring(fd, header_data)
        microrider_config.parse(setupstring)
        # Overwrite any user-defined channel configs
        for channel_config in channel_configs:
            microrider_config.update_config(channel_config)
        # Create the channel matrix
        channel_matrix: ChannelMatrix = ChannelMatrix(microrider_config,
                                                      header_data,
                                                      header_info)

        # Now we are ready to read in the data, which could
        # technically be done in small parts too.
        # Read the data and cast it in a numpy array
        fd.seek(HeaderEnum.HeaderSize + header_data.setupfile_size, 0)
        data_bytes = fd.read()

    data_dict = channel_matrix.convert_channels(data_bytes)
    return data_dict
