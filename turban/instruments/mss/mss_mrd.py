import datetime
import warnings
from pytz import timezone
import math
import numpy
import logging
import sys
import pkg_resources
import os
import hashlib
import re
import glob

from . import mss_utils
import scipy.signal
import errno
import copy
import gsw
import xarray as xr

try:
    import dateparser
except ImportError:
    warnings.warn("Could not import dateparser")
    dateparser = None


# Get the version
version_file = pkg_resources.resource_filename("turban", "VERSION")

with open(version_file) as version_f:
    version = version_f.read().strip()

# Setup logging module
# logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)



def read_mrd(filestream, header_only=False, pos_time_only=False, logger=None):
    """Read a binaray SST MRD (Microstructure Raw Data) file

    Parameters
    ----------
    filestream
    header_only
    pos_time_only

    """
    if logger is None:
        logger = logging.getLogger("turban.instruments.mss_mrd")
        logger.setLevel(logging.DEBUG)

    funcname = "read_mrd():"
    f = filestream

    n = 0
    np = 0  # Number of packets
    end_of_header = 0
    end_of_header_tmp = 0
    IN_HEADER = True
    header = b""
    HAVE_TIME = False
    HAVE_POS = False

    # The data
    data = {"date": [], "gps": []}
    data_tmp = {7: [], 8: []}
    nread = 4096
    bind = 0
    btmp = f.read(nread)
    logger.info(funcname + "Start reading file")
    while True:
        if HAVE_TIME & HAVE_POS & pos_time_only:
            # print('Found time and pos, breaking at',n)
            break

        if len(btmp) < 17:
            btmp = btmp + f.read(nread)

        if IN_HEADER:
            # b = f.read(1)
            b = btmp[0:1]
            btmp = btmp[1:]
            bind += 1
            n += 1
        else:
            b = btmp[0:17]
            btmp = btmp[17:]
            bind += 17
            n += 17
            np += 1  # Number of 17 bytes long data packets (packet type + 8 words (little endian)

        if len(b) == 0:  # Nothing left anymore, break
            break
        else:
            # We are in the data body, lets get our data
            if IN_HEADER == False:
                bword = []
                for i in range(1, 16, 2):  # Reading 16 bytes and treat them as words
                    bword.append(int.from_bytes(b[i : i + 2], byteorder="little"))
                if b[0] == 1:  # Time packet of gps
                    if HAVE_TIME == False:
                        year = int(bword[0])
                        year = bword[0]
                        month = bword[1]
                        day = bword[2]
                        hour = bword[4]
                        minute = bword[5]
                        second = bword[6]
                        date = datetime.datetime(
                            year,
                            month,
                            day,
                            hour,
                            minute,
                            second,
                            tzinfo=timezone("UTC"),
                        )
                        data["date"].append([np, date])
                        HAVE_TIME = True

                elif b[0] == 3:  # Position packet of gps
                    if HAVE_POS == False:
                        # Latitude
                        latint = int.from_bytes(b[1:3], byteorder="little")
                        latsign = (latint & 0x8000) >> 15
                        if latsign == 0:
                            latsign = -1
                        lat = latint & 0x1FFF
                        latdeg = (lat - lat % 100) / 100
                        latmindec = int.from_bytes(b[3:5], byteorder="little")
                        if abs(latmindec) > 0:
                            digits = int(math.log10(latmindec)) + 1
                        else:
                            digits = 0

                        latmin = lat % 100 + latmindec / 10**digits
                        latdec = latsign * (latdeg + latmin / 60)
                        # Longitude
                        lonint = int.from_bytes(b[5:7], byteorder="little")
                        lonsign = (lonint & 0x8000) >> 15
                        if lonsign == 0:
                            lonsign = -1
                        lon = lonint & 0x1FFF
                        londeg = (lon - lon % 100) / 100
                        lonmindec = int.from_bytes(b[7:9], byteorder="little")
                        if abs(lonmindec) > 0:
                            digits = int(math.log10(lonmindec)) + 1
                        else:
                            digits = 0

                        lonmin = lon % 100 + lonmindec / 10**digits
                        londec = lonsign * (londeg + lonmin / 60)
                        # Daytime
                        tmp = (bword[6]) + (bword[4] & 0x000F) * 100000
                        hour = int((tmp - tmp % 10000) / 10000)
                        tmp2 = tmp % 10000
                        minute = int((tmp2 - tmp2 % 100) / 100)
                        second = tmp2 % 100
                        data["gps"].append(
                            [np, londec, latdec, lon, lat, hour, minute, second]
                        )
                        HAVE_POS = True

                elif b[0] == 7:  # Channels 0-7 od ADC
                    bc = numpy.frombuffer(b[1:], dtype=numpy.uint16)
                    data_tmp[7].append(bc)

                elif b[0] == 8:  # Channels 8-15 of ADC
                    bc = numpy.frombuffer(b[1:], dtype=numpy.uint16)
                    data_tmp[8].append(bc)

            else:  # We are in the header
                header += b
                if (
                    b == b"\x1a"
                ):  # Header is filled up with 0x1A until it is dividable by 17
                    end_of_header = n
                    data["end_of_header"] = end_of_header
                    # Check if dividable by 17
                    if (end_of_header) % 17 == 0:  # Dividable by 17
                        data["valid_mrd"] = True
                        # print('Found a valid header (dividable by 17)')
                        try:
                            data["header"] = header.decode("UTF-8")
                        except:
                            logger.info(
                                "UTF-8 did not work, trying to decode using Window charset"
                            )
                            data["header"] = header.decode("Windows-1252")
                        IN_HEADER = False
                        if header_only:
                            return data

    numsamples = min([len(data_tmp[7]), len(data_tmp[8])])
    data["numsamples"] = numsamples
    data["channels"] = numpy.hstack(
        [
            numpy.asarray((data_tmp[7][:numsamples])),
            numpy.asarray((data_tmp[8][:numsamples])),
        ]
    )
    return data


def parse_header(header, logger=None):
    """Parsing the header of a MRD file and saving the results in a config dictionary"""
    if logger is None:
        logger = logging.getLogger("turban.instruments.mss_mrd")
        logger.setLevel(logging.DEBUG)

    funcname = __name__ + ".parse_header():"
    config = {"mss": {"channels": {}}}
    ind_ship = header.find("Ship    :   ") + len("Ship    :   ")
    ship = header[ind_ship : ind_ship + 8]
    ship = ship.rstrip("_")
    config["ship"] = ship

    ind_cruise = header.find("Cruise:") + len("Cruise:")
    cruise = header[ind_cruise + 1 : ind_cruise + 18]
    cruise = cruise.rstrip("_")
    config["cruise"] = cruise

    # hs = header.split('\\r')
    hs = header.splitlines()
    sensor_str = []

    logger.debug("PC-Time line:{}".format(hs[2]))
    config["date_pc"] = dateparser.parse(hs[2]).isoformat()
    # print('Config date',config['date_pc'])
    # print('HS')
    # print('hs',hs)
    # print('HS done')
    # print('Type',type(header))
    # Loop over all channels
    # for i in range(17,len(hs)-1):
    for i in range(len(hs)):
        # logger.debug('Testing line {:s}'.format(hs[i]))
        hstmp = re.sub("\s+", " ", hs[i])  # replace multiple blanks with one
        hsp = hstmp.split(" ")
        # check if we have a sensor line
        # each sensor should have 12 entries
        # print('hsp',hsp,len(hsp))
        if len(hsp) >= 12:
            FLAG_sensor = True
            try:
                devicenum = int(hsp[0].replace(" ", ""))
            except:
                FLAG_sensor = False
            try:
                channelnum = int(hsp[2].replace(" ", ""))
            except:
                FLAG_sensor = False
        else:
            FLAG_sensor = False

        if FLAG_sensor:
            # logger.debug(funcname + 'Found sensor')
            sensor_str.append(hs[i])
            devicename = hsp[1].replace(" ", "")
            if "MSS" in devicename:  # Treat the MSS here
                config["mss"]["name"] = devicename
                config["mss"]["devicenum"] = devicenum
                channelnum = int(hsp[2].replace(" ", ""))
                channel = hsp[4].replace(" ", "")
                unit = hsp[5].replace(" ", "")
                caltype = hsp[3].replace(" ", "")
                config["mss"]["channels"][channelnum] = {}
                config["mss"]["channels"][channelnum]["unit"] = unit
                config["mss"]["channels"][channelnum]["name"] = channel
                config["mss"]["channels"][channelnum]["caltype"] = caltype
                poly = []
                # There are 6 coefficients for each channel
                for val in hsp[6:12]:
                    poly.append(float(val))

                config["mss"]["channels"][channelnum]["coeff"] = numpy.asarray(poly)
                # print(devicename,channelnum,channel)
                # if(hs[i].upper().find('COUNT') >=0):
                #    mss = hs[i].split(' ')[1]

    return config


def raw_to_level0(mss_config, rawdata, logger=None):
    if logger is None:
        logger = logging.getLogger("turban.instruments.mss_mrd")
        logger.setLevel(logging.DEBUG)

    funcname = "raw_to_units():"
    logger.debug(funcname)
    rawdatac = rawdata["channels"]
    # print('rawdata keys',rawdata.keys())
    try:
        gps = rawdata["gps"]
        gps_date = rawdata["date"]
        logger.debug("Found gps information")
    except:
        logger.debug("No gps information in data")
        gps = None
        pass

    count_offset = mss_config.offset
    if rawdata["numsamples"] == 0:
        logger.info(funcname + " No samples found for conversion.")
        return None

    # Create matrix for converted data
    # data = numpy.zeros(numpy.shape(rawdatac)) * numpy.nan
    data = {}
    # Create a xarray
    #
    nsamples = len(rawdatac[:, 0])
    index = numpy.arange(nsamples)
    level0_dataset = xr.Dataset(
        coords={
            "index": index,
        },
    )

    # Add gps information
    if gps:
        level0_dataset.attrs["longitude"] = gps[0][1]
        level0_dataset.attrs["latitude"] = gps[0][2]
        level0_dataset.attrs["date_gps"] = gps_date[0][1].isoformat()
    else:
        logger.warning('No GPS data available, setting longitude/latitude to 0.')
        level0_dataset.attrs["longitude"] = 0
        level0_dataset.attrs["latitude"] = 0
        level0_dataset.attrs["date_gps"] = None
    # Add the header to the dataset
    # try:
    if True:
        headerstr = rawdata["header"]
        header = parse_header(headerstr)
        for k in header.keys():
            if "mss" not in k:  # Ignore the mss part
                level0_dataset.attrs["header_" + k] = str(header[k])
    # except:
    #    logger.debug('Could not add header', exc_info=True)

    tstr = datetime.datetime.now().isoformat()

    level0_dataset.attrs["history"] = "Created with {} (v{}) on the {}".format(
        logger.name, version, tstr
    )

    for s in mss_config.sensors:
        logger.debug("Converting {}".format(s))
        sensor = mss_config.sensors[s]
        i = sensor.channel - 1  # Channels are saved in index notation starting with 1
        caltype = sensor.calibration_type
        coeff = sensor.coefficients
        unit = sensor.unit
        channelname = sensor.name
        offset = mss_config.offset
        data_tmp = sensor.raw_to_units(rawdatac[:, i], offset)
        data[channelname] = data_tmp
        level0_dataset[channelname] = (["index"], data_tmp)
        level0_dataset[channelname].attrs["units"] = unit
        if "SHE" in channelname:
            level0_dataset[channelname].attrs["sensitivity"] = sensor.sensitivity

    # print('Config',self.config)
    # Calculate a time [s]
    # TODO: Use the COUNT variable, this is a stupid counting without checking for missing data
    # count_offset_new = len(data['COUNT']) + count_offset
    # data['COUNT_TOTAL'] = numpy.arange(count_offset, count_offset_new)
    # time_offset = self.config['time_offset_unix']
    # data['t'] = data['COUNT_TOTAL'] / config['fs'] + time_offset
    # return data
    return level0_dataset


def level0_to_level1(mss_config, level0, pspd_rel=None, logger=None):
    if logger is None:
        logger = logging.getLogger("turban.instruments.mss_mrd")
        logger.setLevel(logging.DEBUG)

    funcname = __name__ + ".level0_to_level1():"
    logger.debug(funcname)
    logger.debug("Pressure sensor:{}".format(mss_config.pressure_sensorname))
    logger.debug("Keys of level0 data:{}".format(level0.keys()))
    if True:
        # Loop over all shear sensors and calculate shear
        shearsensors = []
        shear = []
        for k in level0.keys():
            try:
                numshear = int(k[3])
            except:
                numshear = None

            if ("SHE" in k) and (numshear is not None):
                shearsensors.append(k)

        n_shear = range(len(shearsensors))
        index = level0["index"]
        nsamples = len(level0["index"])

        level1_dataset = xr.Dataset(
            coords={
                "n_shear": n_shear,
                "index": index,
            },
        )
        # Count the number of samples and divide by sampling frequency, to get a time vector
        time_count = numpy.cumsum(index) / mss_config.sampling_freq
        print("time count", time_count)
        print("Time count", numpy.shape(time_count), numpy.shape(index))
        channelname = "time_count"
        level1_dataset[channelname] = time_count
        level1_dataset[channelname].attrs["units"] = "s"
        level1_dataset[channelname].attrs[
            "description"
        ] = "count of samples divided by sampling frequency"

        # Create a time vector, TODO: This has to be improved alot!
        if False:  # Calculate time
            dt = 1 / mss_config.sampling_freq
            t = numpy.arange(0, nsamples) * dt
            Dt = t[-1]
            if False:
                level1["TIME"] = t
                tstr = self.meta["date"].strftime(
                    "seconds since %Y-%m-%d %H:%M:%S +0:00"
                )  # TODO allow different time zones
                level1_units["TIME"] = tstr
            else:
                tstr = self.meta["date"].strftime(
                    "seconds since %Y-%m-%d %H:%M:%S +0:00"
                )  # TODO allow different time zones
                level1_units["TIME"] = tstr
                level1["TIME"] = level0["t"]

        # Derive salinity, conservative temperature and density
        cond_sensorname = mss_config.sensornames_ctd["cond"]
        temp_sensorname = mss_config.sensornames_ctd["temp"]
        press_sensorname = mss_config.sensornames_ctd["press"]
        cond = level0[cond_sensorname].copy()
        cond[cond < 0] = 0
        SP = gsw.SP_from_C(cond, level0[temp_sensorname], level0[press_sensorname])
        level1_dataset["PSAL"] = SP
        level1_dataset["PSAL"].attrs["units"] = "1"


        SA = gsw.SA_from_SP(SP, level0[press_sensorname], level0.longitude, level0.latitude)
        level1_dataset["SA"] = SA
        level1_dataset["SA"].attrs["units"] = "g kg-1"
        # Calculating conservative temperature
        CT = gsw.CT_from_t(SA, level0[temp_sensorname], level0[press_sensorname])
        level1_dataset["CT"] = CT
        level1_dataset["CT"].attrs["units"] = "degC"
        dens = gsw.rho(SA, CT, level0[press_sensorname])
        level1_dataset["DENS"] = dens
        level1_dataset["DENS"].attrs["units"] = "kg m-3"
        # Calculating sinking velocity
        try:
            config_pspd_rel = self.config["pspd_rel"]
            config_pspd_rel_data = self.config["pspd_rel_data"]
        except Exception as e:
            # self.logger.exception(e)
            config_pspd_rel = None

        # print('config', self.config)
        # print('config2', config_pspd_rel,config_pspd_rel_data)
        # if self.config['mss']['pspd_rel'] = 'external'
        if mss_config.pspd_rel_method == "constant":
            logger.debug(
                funcname
                + "Using constant velocity {:f}".format(
                    mss_config.pspd_rel_constant_vel
                )
            )
            vsink = numpy.zeros(nsamples) + mss_config.pspd_rel_constant_vel
        elif (pspd_rel is None) or (mss_config.pspd_rel_method == "pressure"):
            logger.debug(funcname + "Using change of pressure to caluclate velocity")
            vsink = mss_utils.calc_vsink(
                press=level0[press_sensorname], fs=mss_config.sampling_freq
            )
        elif mss_config.pspd_rel_method == "external":
            logger.debug(funcname + "Using external")
            vsink = pspd_rel
        else:
            self.logger.warning(funcname + " no method to get velocity past sensor")
            raise ValueError

        level1_dataset["PSPD_REL"] = (["index"], vsink)
        level1_dataset["PSPD_REL"].attrs["units"] = "m s-1"

        # copy some variables, TODO: discuss variablenames
        for sensor in [press_sensorname, temp_sensorname, cond_sensorname]:
            level1_dataset[sensor] = level0[sensor]
            level1_dataset[sensor].attrs["units"] = level0[sensor].attrs["units"]

        # Loop over all shear sensors and calculate shear
        shearsensors = []
        shear = []
        for k in level0.keys():
            try:
                numshear = int(k[3])
            except:
                numshear = None

            if ("SHE" in k) and (numshear is not None):
                shearsensors.append(k)
                # print('Found shear sensor {:s}, calculating shear'.format(k))
                SH = mss_utils.calc_shear(
                    level0[k], vsink, dens, fs=mss_config.sampling_freq
                )
                shear.append(SH)

        shear = numpy.asarray(shear)
        shear[numpy.isinf(shear)] = numpy.nan
        level1_dataset["SHEAR"] = (["n_shear", "index"], shear)

        return level1_dataset

