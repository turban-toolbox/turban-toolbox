import os
import pytest

import logging

from turban.utils.logging import LoggerConfig, LoggerManager



def parse_string(s):
    fields = [i.strip() for i in s.strip().split("|")]
    return fields

@pytest.fixture
def logger_data():
    fn = "mylogger.txt"
    os.remove(fn) if os.path.exists(fn) else None
    handler = logging.FileHandler(fn)
    formatter = logging.Formatter("{levelname} | {name} | {funcName} | {message}",
                                  style="{",
                                  )

    config = LoggerConfig(handler, formatter)
    logger_manager = LoggerManager()
    return logger_manager, config, fn
    

def test_LoggerManager(logger_data):
    logger_manager, config, fn = logger_data
    assert os.path.exists(fn)
    os.remove(fn)


def test_create_single_logger(logger_data):
    logger_manager, config, fn = logger_data
    logger = logger_manager.get_logger("single", config)
    logger.warning("warning message")
    with open(fn) as fp:
        lines = fp.readlines()
    assert len(lines)==1
    fields = parse_string(lines[0])
    assert fields[0] == "WARNING"
    assert fields[1] == "single"
    assert fields[3] == "warning message"
    os.remove(fn)

def test_create_double_logger_selective_levels(logger_data):
    logger_manager, config, fn = logger_data
    logger0 = logger_manager.get_logger("first", config)
    logger1 = logger_manager.get_logger("second", config)
    logger_manager.set_level("debug", "second")
    
    logger0.warning("warning message") # should print
    logger1.warning("warning message") # should print
    logger0.debug("debug message") # should NOT print
    logger1.debug("debug message") # should print
    with open(fn) as fp:
        lines = fp.readlines()
    assert len(lines)==3
    fields = parse_string(lines[0])
    assert fields[0] == "WARNING"
    assert fields[1] == "first"
    assert fields[3] == "warning message"
    fields = parse_string(lines[1])
    assert fields[0] == "WARNING"
    assert fields[1] == "second"
    assert fields[3] == "warning message"
    fields = parse_string(lines[2])
    assert fields[0] == "DEBUG"
    assert fields[1] == "second"
    assert fields[3] == "debug message"
    os.remove(fn)

