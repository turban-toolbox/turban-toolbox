import enum

from functools import partial
import io
import logging
import os.path
import re
import struct
from typing import Any, Self

import arrow
import numpy as np
import numpy.typing as np_typing
from scipy.interpolate import interp1d as si_interp1d

from . import  rsConversions
from . import rsConfig_parser
from .rsCommon import ByteHeader, Header, ChannelConfigABC, ChannelConfig, channel_config_factory

logger = logging.getLogger(__name__)




class HeaderEnum(enum.IntEnum):
    HeaderSize = 128
    WordSize = 2
    HeaderFields = HeaderSize // WordSize
    Endianness = 63
    


class uriderException(Exception):
    pass




    
        
class Channel(object):
    '''Channel object

    Parameters
    ----------
    name : string
        name of channel
    config : ChannelConfig
        channel configuration object (from setupstr)
    deconvolved : bool (False)
        flag to indicate whether this is a deconvolved channel.

    This class holds the data of each channel:
    when read     : the raw data are in .data (._data is None)
    when converted: the converted data are in .data and the raw data in ._data.

    The following methods are provided:

    copy(): this returns a new channel, but keeps the same configuration, 
            and sets the deconvolved flag. This method is used to create 
            new channels for the high resolution parameters

    correct_sign(): sign correction (u2 -> i2) if configuration dictates.

    convert_to_units(): applies to associated conversion algortithm (determined 
           from the config dictionary) to the raw data.

    join(): joins even and odd channels. Warning not fully tested. This method
           can be called by through the + operator: ch = chE+chO
           
   
    '''

     
    def __init__(self,
                 channel_config: ChannelConfigABC,
                 deconvolved: bool = False):
        self.name: str = channel_config.name
        self.config: ChannelConfigABC = channel_config
        self.data: np.typing.NDArray[np.float64] = np.array([])
        self._data: np.typing.NDArray[np.float64] = np.array([])
        self.converter = rsConversions.get_converter(channel_config)(channel_config)
        self.converted_into_units: bool = False
        self.deconvolved: bool = deconvolved
        
    def __add__(self, rhs: "Channel") -> "Channel":
        return self.join(self, rhs)

    def copy(self, new_name:str = "", deconvolved: bool = True) -> "Channel":
        ''' Copies configuration into a new channel with optionally a new name

        Parameters
        ----------
        new_name : string (None)
            gives this name to the new channel. If None, it keeps the original name.
        
        Returns
        -------
        Channel object
            Empty channel, with the configuration copied, and the deconvolved flag set.
        '''
        name = new_name or self.name
        config = self.config.copy()
        return Channel(config, deconvolved=deconvolved)
    
    def correct_sign(self) -> None:
        '''Corrects sign of data (if configuration dictates)
        
        Some channels record in unsigned integers. If required they need to be 
        signed, where the msb is used as sign.

        '''
        types_to_skip = "sbt sbc jac_c jac_t o2_43f".split()
        cfg = self.config
        if cfg.id == 255:
            return # nothing to do for ch255
        if cfg.type in types_to_skip:
            return
        if cfg.sign == 'unsigned':
            logger.info("Correcting sign")
            self.data = self.data.astype(np.dtype(f'>u{HeaderEnum.WordSize}')) # as unsigned
        else:
            idx = np.where(self.data>=2**31)[0]
            if len(idx):
                self.data[idx] -= 2**32
                logger.info("Correcting sign of 32 bit channel.")
                logger.debug("Does this ever happen?")
                raise ValueError("FIX ME")

    def convert_to_units(self) -> None:
        '''Converts channel into unit bearing data
        
        The attribute .data is copied to ._data (raw data) and
        converted using a conversion method associated with the
        type field of the configuration.

        Sets the converted_into_units flag.
        '''
        if not self.converted_into_units:
            self._data = self.data
            self.data = self.converter(self._data)
            self.converted_into_units = True
        else:
            raise ValueError('Refusing to convert to units for a parameter already converted.')


        
        
                         
    def join(self, even_channel: "Channel", odd_channel: "Channel") -> "Channel" :
        '''Joins even and odd channel

        Parameters
        ---------
        even_channel : Channel
            Channel with the even signal (16 bit integer)
        odd_channel : Channel
            Channel with the odd signal (16 bit integer)

        Returns
        -------
        Channel
            Channel with combined channel (32 bit)

        Note: Not fully tested.
        '''
        chE = even_channel.data.astype(np.dtype(f'>u{HeaderEnum.WordSize}')) # as unsigned.
        chO = odd_channel.data.astype(np.dtype(f'>u{HeaderEnum.WordSize}')) # as unsigned
        chEO = np.zeros(chE.shape, dtype=f'>u{HeaderEnum.WordSize*2}') # unsigned, but 32 bit
        chEO[...] = chO*2**16 + chE # because chO is 16 bit we cannot shift 16 bits...
        even_channel.config.name = even_channel.name.replace('_E','')
        channel = Channel(even_channel.config)
        channel.data = chEO
        return channel


class ChannelMatrix(object):
    ''' Data class to store the sensor matrix
    
    Parameters
    ----------
    microrider_config : dict[str, str|int|float|list[int]]
        dictionary holding the MR configuration

    '''
    def __init__(self, microrider_config: rsConfig_parser.MicroRiderConfig) -> None:
        self.matrix = self._get_matrix(microrider_config)
        self.number_of_elements = np.prod(self.matrix.shape)
        self.channels = self._create_channels(microrider_config)
        
    def _get_matrix(self, microrider_config: rsConfig_parser.MicroRiderConfig) -> np_typing.NDArray[np.float64]:
        matrix = []
        matrix_section = microrider_config.get_section('matrix')
        # we don't always have the number of rows given anymore.
        # Find out ourselves
        n = 0
        while True:
            k = f"row{n+1:02d}" # Here they count from 1, for variation I suppose.
            if not k in matrix_section:
                break
            s = matrix_section[k]
            matrix.append(s)
            n+=1
        logger.info("Address matrix:")
        for row in matrix:
            _s = " ".join([f"{i:3d}" for i in list(row)])
            logger.info(f"| {_s} |")
        return np.array(matrix)

    def _create_channels(self, microrider_config: rsConfig_parser.MicroRiderConfig) -> dict[str, Channel]:
        """
        """
        channels: dict[str, Channel] = {}

        logger.info("Available channels:")
        for n in range(microrider_config.number_of_channels):
            section_name = f"channel{n:02d}"
            section_dict = microrider_config.get_section(section_name)
            channel_config = self._create_channel_config(section_dict)
            channel_name = channel_config.name
            channels[channel_name] = Channel(channel_config)
            logger.info(f"\t{n:2d}: {channel_name}")
        if np.any(self.matrix==255):
            logger.debug("Created Channel 255...")
            channel_config = ChannelConfig(name="ch255", id=255, type="")
            channels["ch255"] = Channel(channel_config)
            n+=1
            logger.info(f"\t{n:2d}: ch255")
        return channels

    def _create_channel_config(self, section : dict[str, Any]) -> ChannelConfigABC:
        name = section["name"]
        _ChannelConfig = channel_config_factory(name)
        channel_config = _ChannelConfig(name)
        for k_any_case, v in section.items():
            k = k_any_case.lower()
            if k == "name":
                continue
            channel_config.update(k, v)
        return channel_config
    #
    #  This block below codes for split channels. We seem not to have them.
    #  So we cannot test this. See also combine_split_channels() method
    #
    
    
        # for k, v in self.config.cfg.items():
        #     # a channel has an id, name and type
        #     try:
        #         channel_id = v['id']
        #         channel_name = v['name']
        #         channel_type = v['type']
        #     except KeyError:
        #         continue
        #     if type(channel_id) == int:
        #         channels[channel_name] = Channel(channel_name, v)
        #         logger.info(f'     channel : {channel_id:02d} = {channel_name}') 
        #     else:
        #         # accommodate for "3,4" "3 4" and "3, 4" and even ";" as separator
        #         s = channel_id.replace(",", " ")
        #         flds = s.split()
        #         channel_id_even, channel_id_odd = [int(i) for i in flds]
        #         config_even = v.copy()
        #         config_even['id'] = channel_id_even
        #         config_odd = v.copy()
        #         config_odd['id'] = channel_id_odd
        #         channels[channel_name+"_E"] = Channel(channel_name, config_even)
        #         channels[channel_name+"_O"] = Channel(channel_name, config_odd)
        #         logger.info(f'even channel : {channel_id_even:02d} = {channel_name}') 
        #         logger.info(f' odd channel : {channel_id_odd:02d} = {channel_name}') 
        # # From the original code:
        # # The special characeter is not listed as a channel. Insert it manually if it exists within the channel matrix.
        # if np.any(self.channel_matrix.matrix==255):
        #     channels["ch255"] = Channel("ch255", dict(id=255))
        # return channels
            



        
        
class HeaderParser():
    def parse(self, fd: io.BufferedReader) -> ByteHeader:
        '''Parses header block of a data record. It is assumed that the block is HeaderEnum.HeaderSize large.

        Parameters
        ----------
        fd : io.BufferedReader
            file descriptor pointing to an open file

        Returns
        -------
        ByteHeader:
            a dataclass with (byte) header information
        '''
        block = fd.read(HeaderEnum.HeaderSize)
        match self.__get(HeaderEnum.Endianness, block=block, fmt="<H"):
            case 0:
                get = partial(self.__get, block=block, fmt="<H")
            case 1 | 256: # little endian
                get = partial(self.__get, block=block, fmt="<H")
            case 2 | 512:  # big endian
                get = partial(self.__get, block=block, fmt=">H")
            case _:
                raise ValueError("Failed to determine the endianness of this block of data.")
                
        # From the manual...
        byte_header = ByteHeader(file_number = get(0),
                                 record_number = get(1),
                                 record_number_serial_port = get(2),
                                 year = get(3),
                                 month = get(4),
                                 day = get(5),
                                 hour = get(6),
                                 minute = get(7),
                                 second = get(8),
                                 millisecond = get(9),
                                 header_version = float(get(10)>>8) + float((get(10) & 0x0f))/1000,
                                 setupfile_size = get(11),
                                 product_ID = get(12),
                                 build_number = get(13),
                                 timezone_in_minutes = get(14),
                                 buffer_status = get(15),
                                 restarted = get(16),
                                 record_header_size = get(17) //  HeaderEnum.WordSize,
                                 data_record_size = get(18) //  HeaderEnum.WordSize,
                                 number_of_records_written = get(19),
                                 frequency_clock = float(get(20)) + float(get(21))/1e3,
                                 fast_cols = get(28),
                                 slow_cols = get(29),
                                 n_rows = get(30),
                                 data_size = (get(18) - get(17)) //  HeaderEnum.WordSize
                                 )
        return byte_header

    def __get(self, pos: int, block: bytes=b'', fmt:str="<H") -> int:
        p = pos*HeaderEnum.WordSize
        v = int(struct.unpack(fmt, block[p:p+HeaderEnum.WordSize])[0])
        return v

    def check_for_bad_blocks(self, fp: io.BufferedReader) -> int:
        '''Checks for bad blocks

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

        '''
        current_fp_location: int = fp.tell()
        fp.seek(0, 2) # go to the end
        file_size: int = fp.tell()
        fp.seek(0, 0) # rewind
        record_number: int = 0
        bytes_read: int = 0
        while True:
            # we are going to read 128 bytes. Make sure that that is possible
            if bytes_read == file_size:
                break # we are done
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
        fp.seek(current_fp_location, 0) # go back to where we started.
        record_number -= 1 # We count the first record too, but this is not a data record.
        return record_number

    def read_setupstring(self,
                         fd : io.BufferedReader,
                         byte_header: ByteHeader) -> str:
        ''' Reads the setupstring from the binary file
        
        Parameters
        ----------
        fd : file descriptor
        
        byte_header: ByteHeader
            ByteHeader dataclass

        Returns
        -------
        str
            setupstring.

        '''
        fd_current = fd.tell()
        fd.seek(HeaderEnum.HeaderSize, 0)
        setupfile_size = byte_header.setupfile_size
        bytestr = fd.read(setupfile_size)
        s = bytestr.decode()
        fd.seek(fd_current, 0)
        return s


class MicroRiderData(object):
    ''' Data class to hold all data collected by the RSI MicroRider.

    parameters
    ----------

    path : string
        full path to .P file

    config: system configuration, extracted from the setupfile string.

    '''
    def __init__(self, path: str, config: rsConfig_parser.MicroRiderConfig) -> None:
        self.config: rsConfig_parser.MicroRiderConfig = config
        self.header : Header
        self.channel_matrix: ChannelMatrix = ChannelMatrix(config)
        self._regex_deconvolve_preemph_name = re.compile(r"([TP].?)_d([TP].?)")
        self._regex_deconvolve_name = re.compile(r"([TP]\d?)")
        self._path = path

    def __getattr__(self, name: str) -> Channel:
        try:
            return self.channel_matrix.channels[name]
        except KeyError:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def keys(self) -> list[str]:
        return list(self.channel_matrix.channels.keys())

        
    def add_header_data(self, byte_header: ByteHeader, n_records : int) -> None:
        '''Adds specific parameters from the byteheader to the data structure
        
        Parameters
        ----------
        byte_header : ByteHeader
            ByteHeader dataclass
        '''
        n_cols = byte_header.fast_cols + byte_header.slow_cols
        matrix_count = n_records * byte_header.data_size // (byte_header.n_rows * n_cols)
        fs_fast = byte_header.frequency_clock / n_cols
        fs_slow = byte_header.frequency_clock / n_cols / byte_header.n_rows
        header_version = byte_header.header_version
        t_slow = np.arange(matrix_count) / fs_slow
        t_fast = np.arange(matrix_count*byte_header.n_rows) / fs_fast
        timestamp, datestring, timestring = self.get_file_timestamp(byte_header)
        
        self.header = Header(full_path = os.path.realpath(self._path),
                             n_cols = n_cols,
                             n_records = n_records,
                             fs_fast = fs_fast,
                             fs_slow = fs_slow,
                             header_version = header_version,
                             matrix_count = matrix_count,
                             t_fast = t_fast,
                             t_slow = t_slow,
                             timestamp = timestamp,
                             date = datestring,
                             time = timestring)

        
    def get_file_timestamp(self, byte_header: ByteHeader) -> tuple[float, str, str]:
        '''Gets time stamp of the file
        
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

        '''
        date = arrow.get(byte_header.year,
                         byte_header.month,
                         byte_header.day,
                         byte_header.hour,
                         byte_header.minute,
                         byte_header.second,
                         byte_header.millisecond)
        timestamp = date.timestamp();
        datestr = date.strftime("%Y-%m-%d")
        timestr = date.strftime("%H-%M-%S")
        return timestamp, datestr, timestr
        
        

    def add_channel_data(self, data_snr : np_typing.NDArray[np.float64]) -> None:
        ''' Adds the channel data, from array with dtype i2/u2

        Parameters
        ----------

        data_snr : array 
            array with bytes converted to signed and unsigned 16 bit integers.

        The method adds all the numeric data conveyed in this matrix to the specific
        channels. It uses the sensor matrix to determine which part goes to what 
        channel. 

        Note: this channel adds a keyword to the configuration: depending on the 
              number of values per record, it adds whether this is a fast or
              slow senor if the frequency is 512 or 64, respectively.
        '''
        n_matrix_rows, _ = self.channel_matrix.matrix.shape
        channel_positions = self.channel_matrix.matrix.flatten()
        for channel_name, channel in self.channel_matrix.channels.items():
            logger.debug(f"Channel name : {channel_name}")
            i_channel = channel.config.id
            idx = np.where(channel_positions==i_channel)[0]
            n_data_per_record = len(idx)
            if n_data_per_record == n_matrix_rows:
                channel.config.update('sample_rate', self.header.fs_fast)
            elif n_data_per_record == 1:
                channel.config.update('sample_rate', self.header.fs_slow)
            else:
                channel.config.update('sample_rate', 0) # we don't need to know this for this channel anyway.
            channel.data = data_snr[:, idx].flatten()
            # correct the sign of the data if necessary
            channel.correct_sign()

    def combine_split_channels(self) -> None:
        ''' Combines even and odd channels into one

        All channels that are labelled *_E and *_O are combined into one, using 32 bit ints.
        
        Note: This method has not thorougly been tested because the test data don't have any 
              split channels. It was tested modifying the data file, which has been reverted 
              back again.
        '''
        channels = self.channel_matrix.channels
        even_channels = [ch for ch in channels.keys() if ch.endswith("_E")]
        for ch_even in even_channels:
            ch_odd = ch_even.replace('_E', '_O')
            ch = ch_even.replace('_E', '')
            if not ch_odd in channels.keys():
                logger.warning(f"Found channel {ch_even}, but did not find {ch_odd}")
                continue # No odd channel found. 
            self.channel_matrix.channels[ch] = channels[ch_even] + channels[ch_odd]
            self.channel_matrix.channels.pop(ch_even)
            self.channel_matrix.channels.pop(ch_odd)
            

    def convert_channels(self) -> None:
        ''' Converts all channels

        Converts all channels to data with physical units. If a channel
        needs to be deconvolved first, this is also done (and leads to
        a new channel).
        '''
        for n, ch in self.channel_matrix.channels.copy().items():
            regex_preemph = self._regex_deconvolve_preemph_name.match(n)
            regex = self._regex_deconvolve_name.match(n)
            if regex_preemph:
                # These channels need first to be deconvolved.
                logger.info(f"Deconvolving and converting {n}")
                self.deconvolve_and_convert_channel(ch)
            elif not regex: # skipping T1 T2 and P, because they are interpolated from the deconvolved versions
                logger.info(f"Converting {n}")
                ch.convert_to_units()


    def deconvolve_and_convert_channel(self, ch: Channel) -> None:
        ''' Treats channels that need deconvolving separately

        Note: this method creates a new channel, with the name
              based on the primary channel (e.g. T1) with "_hires" added.


        '''
        channel_name = ch.name
        co_channel_name, _ = ch.name.split("_")
        co_channel = self.channel_matrix.channels[co_channel_name]
        # this channel should not be converted yet.
        if ch.converted_into_units:
            raise ValueError("This preemph channel is converted to units, but should be deconvolved first.")
        if co_channel.converted_into_units:
            raise ValueError("The channel (without preemphasis) is converted to units, but should be usedin the deconvolution first.")
        logger.debug(f"Trying to deconvolve {channel_name} ({co_channel_name})")
        if not np.isclose(ch.config.sample_rate, self.header.fs_fast):
            ch_fast = self.interpolate_onto_fast_channel(ch)
        else:
            ch_fast = ch
        if not np.isclose(co_channel.config.sample_rate, self.header.fs_fast):
            co_channel_fast = self.interpolate_onto_fast_channel(co_channel)
        else:
            co_channel_fast = co_channel
        X = co_channel_fast.data  # not converted yet, so OK.
        X_dX = ch_fast.data
        if hasattr(ch_fast.config, "diff_gain"):
            diff_gain = ch_fast.config.diff_gain # <- is required to be present
        else:
            raise AttributeError("diff_gain is not an attribute for this configuration object.")
        fs = self.header.fs_fast
        
        new_channel_name = f"{co_channel_name}_hires"
        new_channel = co_channel.copy(new_channel_name) # copy config from T1, name it T1_res
        new_channel.data = rsConversions.Deconvolve(X_dX, X, fs, diff_gain).X_hires
        new_channel.convert_to_units()
        self.channel_matrix.channels[new_channel_name] = new_channel
        # interpolate co channel.
        logger.info(f"Converting {co_channel_name} by means of interpolation of unit-bearing hires channel")
        self.interpolate_co_channel(co_channel, new_channel)
        

    def interpolate_onto_fast_channel(self, ch: Channel) -> Channel:
        logger.debug(f"Interpolating {ch.name} to a fast channel...")
        ch_hires = ch.copy("dummy")
        ch_hires.config.sample_rate = self.header.fs_fast
        ifun = si_interp1d(self.header.t_slow, ch.data, kind="cubic", fill_value="extrapolate")
        ch_hires.data = ifun(self.header.t_fast).astype(np.float64)
        return ch_hires
        
    def interpolate_co_channel(self, co_channel: Channel, hires_channel: Channel) -> None:
        if np.isclose(co_channel.config.sample_rate , self.header.fs_slow):
            tm_i = self.header.t_slow
            ifun = si_interp1d(self.header.t_fast, hires_channel.data, kind="cubic", fill_value="extrapolate")
            co_channel.data = ifun(tm_i).astype(np.float64)
            co_channel.converted_into_units = True
        else:
            raise ValueError("Trying to interpolate to hires from a hires vector. Fix me.")


def read_p_file(filename: str, setupstring_filename: str = "") -> MicroRiderData:
    ''' Function to read a single .p file.

    Parameters
    ----------
    filename : str
         Name of .p file to read
    setupstring_filename : str (Optional : "")
         Name of external setup file to be used.

    Returns
    -------
    MicroRiderData object
        Raw data, converted into physical units.
    '''
    header_parser = HeaderParser()
    microrider_config = rsConfig_parser.MicroRiderConfig()

    full_path = os.path.realpath(filename)

    if setupstring_filename:
        with open(setupstring_filename, "r") as fd:
            setupstring = fd.read()
    else:
        setupstring = ""
        
    with open(filename, 'rb') as fd:
        header_parser = HeaderParser()
        header_data = header_parser.parse(fd)
        n_records = header_parser.check_for_bad_blocks(fd)
        if not setupstring: # not set by external file
            # read embedded setupstring (leaves the fp in the correct place)
            setupstring = header_parser.read_setupstring(fd, header_data)
        microrider_config.parse(setupstring)
        data = MicroRiderData(full_path, microrider_config)
        data.add_header_data(header_data, n_records)
        fd.seek(HeaderEnum.HeaderSize + header_data.setupfile_size, 0)

        # Data to be interpreted as signed integers, header data as unsigned integers.
        dt_data = np.dtype(f'>i{HeaderEnum.WordSize}')
        dt_hdr = np.dtype(f'>u{HeaderEnum.WordSize}')

        # read the data and cast it in a numpy array
        bytes_data = fd.read()
        bindata = np.frombuffer(bytes_data, dtype=dt_data).reshape(-1, header_data.data_record_size)
        # Extract header and data
    data_hdr = bindata[:, :header_data.record_header_size].astype(dt_hdr) # recast as unsigned
    n = data.channel_matrix.number_of_elements
    data_snr = bindata[:, header_data.record_header_size:].reshape(-1, n)
    data.add_channel_data(data_snr)
    #data.combine_split_channels() # <- we don't have split channels. Cannot test.
    data.convert_channels()
    return data
