from itertools import chain
import os

import numpy as np
import pytest
from scipy.io import loadmat

from turban.instruments.RocklandScientific import rsIO, rsConfig_parser, rsCommon

def asDict(mat_variable):
    d = dict([(k,v) for k,v in zip(mat_variable.dtype.names, mat_variable.tolist())])
    return d

datadir = "data/RocklandScientific"

def test_header_data_058():
    header_parser = rsIO.HeaderParser()
    with open(os.path.join(datadir, "DAT_058.P"), "rb") as fp:
        data = header_parser.parse(fp)
    assert \
        data.file_number == 58 and \
        data.year == 2018 and \
        data.data_record_size == 4672 and \
        data.data_size == 4608

def test_header_data_0413():
    header_parser =rsIO.HeaderParser()
    with open(os.path.join(datadir, "data_0413.p"), "rb") as fp:
        data = header_parser.parse(fp)
    assert \
        data.file_number == 413 and \
        data.year == 2025 and \
        data.data_record_size == 4672 and \
        data.data_size == 4608
    
def test_check_for_bad_blocks_058():
    header_parser =rsIO.HeaderParser()
    with open(os.path.join(datadir, "DAT_058.P"), "rb") as fp:
        number_of_records = header_parser.check_for_bad_blocks(fp)
    assert number_of_records == 11593

def test_check_for_bad_blocks_0413():
    header_parser =rsIO.HeaderParser()
    with open(os.path.join(datadir, "data_0413.p"), "rb") as fp:
        number_of_records = header_parser.check_for_bad_blocks(fp)
    assert number_of_records == 483

#
# Test the ChannelMatrix class
#

@pytest.fixture
def microrider_config_data(request):
    variable = request.param
    match variable:
        case "058":
            fn = os.path.join(datadir, "setupstring_058.txt")
        case "0413":
            fn = os.path.join(datadir, "setupstring_0413.txt")
    with open(fn, "r") as fp:
        setupstring = "\n".join(fp.readlines())
    microrider_config = rsConfig_parser.MicroRiderConfig()
    microrider_config.parse(setupstring)
    return dict(microrider_config=microrider_config, hash_value=None)

@pytest.mark.parametrize("microrider_config_data", ["058", "0413"], indirect=True)
def test_channel_matrix(microrider_config_data):
    channel_matrix = rsIO.ChannelMatrix(microrider_config_data["microrider_config"])
    assert channel_matrix.number_of_elements == 72


def test_channel_config():
    channel_config = rsCommon.ChannelConfig(name="ch255", id=255, type="")
    assert channel_config.name == "ch255"
    assert channel_config.id == 255
    assert channel_config.type == ""
    channel_config.update("name", "CH255")
    assert channel_config.name == "CH255"
    with pytest.raises(AttributeError, match="ChannelConfig has no attribute not_exsting."):
        channel_config.update("not_exsting", 13)
        
def test_channel_config_is_set():
    channel_config = rsCommon.ChannelConfigThermistor(name="T1", id=12, type="therm")
    assert channel_config.is_set("name")
    assert not channel_config.is_set("a")
    with pytest.raises(AttributeError, match="ChannelConfigThermistor has no attribute a0"):
        channel_config.is_set("a0")
    channel_config.update("a", 1)
    assert channel_config.is_set("a")



#
# Test MicroRiderData class
#

@pytest.fixture
def microrider_data(request):
    variable = request.param
    match variable:
        case "058":
            fn = os.path.join(datadir ,"DAT_058.P")
            fns = os.path.join(datadir, "setupstring_058.txt")
            # The values below from the odas library
            expected = dict(length_t_fast=5935616,
                            t_fast_max=11592.255,
                            length_t_slow=741952,
                            t_slow_max=11592.242)

        case "0413":
            fn = os.path.join(datadir, "data_0413.p")
            fns = os.path.join(datadir, "setupstring_0413.txt")
            # The values below from the odas library
            expected = dict(length_t_fast=247296,
                            length_t_slow=30912,
                            t_slow_max=482.95,
                            t_fast_max=482.97)
    with open(fns, "r") as fp:
        setupstring = "\n".join(fp.readlines())
    microrider_config = rsConfig_parser.MicroRiderConfig()
    microrider_config.parse(setupstring)
    full_path = os.path.realpath(fn)
    data = rsIO.MicroRiderData(full_path, microrider_config)
    header_parser = rsIO.HeaderParser()
    with open(fn, 'rb') as fd:
        header_data = header_parser.parse(fd)
        n_records = header_parser.check_for_bad_blocks(fd)
        data.add_header_data(header_data, n_records)
    return dict(microrider_config=microrider_config,
                full_path = full_path,
                data = data,
                expected = expected)


@pytest.mark.parametrize("microrider_data", ["058", "0413"], indirect=True)
def test_microriderdata_header(microrider_data):
    microrider_config = microrider_data["microrider_config"]
    data = microrider_data["data"]
    expected = microrider_data["expected"]
    assert np.abs(data.header.t_fast.max() - expected['t_fast_max'])<0.1
    assert len(data.header.t_fast) == expected['length_t_fast']
    assert np.abs(data.header.t_slow.max() - expected['t_slow_max'])<0.1
    assert len(data.header.t_slow) == expected['length_t_slow']

