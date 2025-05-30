import datetime
from pytz import timezone
import math
import numpy
import numpy as np
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
import dateparser

# Get the version
version_file = pkg_resources.resource_filename("turban", "VERSION")

with open(version_file) as version_f:
    version = version_f.read().strip()

# Setup logging module
logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
logger = logging.getLogger("turban.instruments.mss_mrd")
logger.setLevel(logging.DEBUG)


def read_mrd(filestream, header_only=False, pos_time_only=False):
    """ Read a binaray SST MRD (Microstructure Raw Data) file

    Parameters
    ----------
    filestream
    header_only
    pos_time_only

    """
    funcname = 'read_mrd():'
    f = filestream

    n = 0
    np = 0  # Number of packets
    end_of_header = 0
    end_of_header_tmp = 0
    IN_HEADER = True
    header = b''
    HAVE_TIME = False
    HAVE_POS = False

    # The data
    data = {'date': [], 'gps': []}
    data_tmp = {7: [], 8: []}
    nread = 4096
    bind = 0
    btmp = f.read(nread)
    logger.info(funcname + 'Start reading file')
    while True:
        if HAVE_TIME & HAVE_POS & pos_time_only:
            # print('Found time and pos, breaking at',n)
            break

        if (len(btmp) < 17):
            btmp = btmp + f.read(nread)

        if (IN_HEADER):
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
            if (IN_HEADER == False):
                bword = []
                for i in range(1, 16, 2):  # Reading 16 bytes and treat them as words
                    bword.append(int.from_bytes(b[i:i + 2], byteorder='little'))
                if (b[0] == 1):  # Time packet of gps
                    if (HAVE_TIME == False):
                        year = int(bword[0])
                        year = bword[0]
                        month = bword[1]
                        day = bword[2]
                        hour = bword[4]
                        minute = bword[5]
                        second = bword[6]
                        date = datetime.datetime(year, month, day, hour, minute, second, tzinfo=timezone('UTC'))
                        data['date'].append([np, date])
                        HAVE_TIME = True

                elif (b[0] == 3):  # Position packet of gps
                    if (HAVE_POS == False):
                        # Latitude
                        latint = int.from_bytes(b[1:3], byteorder='little')
                        latsign = (latint & 0x8000) >> 15
                        if (latsign == 0):
                            latsign = -1
                        lat = latint & 0x1FFF
                        latdeg = (lat - lat % 100) / 100
                        latmindec = int.from_bytes(b[3:5], byteorder='little')
                        if (abs(latmindec) > 0):
                            digits = int(math.log10(latmindec)) + 1
                        else:
                            digits = 0

                        latmin = lat % 100 + latmindec / 10 ** digits
                        latdec = latsign * (latdeg + latmin / 60)
                        # Longitude
                        lonint = int.from_bytes(b[5:7], byteorder='little')
                        lonsign = (lonint & 0x8000) >> 15
                        if (lonsign == 0):
                            lonsign = -1
                        lon = lonint & 0x1FFF
                        londeg = (lon - lon % 100) / 100
                        lonmindec = int.from_bytes(b[7:9], byteorder='little')
                        if (abs(lonmindec) > 0):
                            digits = int(math.log10(lonmindec)) + 1
                        else:
                            digits = 0

                        lonmin = lon % 100 + lonmindec / 10 ** digits
                        londec = lonsign * (londeg + lonmin / 60)
                        # Daytime
                        tmp = (bword[6]) + (bword[4] & 0x000F) * 100000
                        hour = int((tmp - tmp % 10000) / 10000)
                        tmp2 = tmp % 10000
                        minute = int((tmp2 - tmp2 % 100) / 100)
                        second = tmp2 % 100
                        data['gps'].append([np, londec, latdec, lon, lat, hour, minute, second])
                        HAVE_POS = True

                elif (b[0] == 7):  # Channels 0-7 od ADC
                    bc = numpy.frombuffer(b[1:], dtype=numpy.uint16)
                    data_tmp[7].append(bc)

                elif (b[0] == 8):  # Channels 8-15 of ADC
                    bc = numpy.frombuffer(b[1:], dtype=numpy.uint16)
                    data_tmp[8].append(bc)


            else:  # We are in the header
                header += b
                if (b == b'\x1A'):  # Header is filled up with 0x1A until it is dividable by 17
                    end_of_header = n
                    data['end_of_header'] = end_of_header
                    # Check if dividable by 17
                    if ((end_of_header) % 17 == 0):  # Dividable by 17
                        data['valid_mrd'] = True
                        # print('Found a valid header (dividable by 17)')
                        try:
                            data['header'] = header.decode('UTF-8')
                        except:
                            logger.info('UTF-8 did not work, trying to decode using Window charset')
                            data['header'] = header.decode('Windows-1252')
                        IN_HEADER = False
                        if header_only:
                            return data

    numsamples = min([len(data_tmp[7]), len(data_tmp[8])])
    data['numsamples'] = numsamples
    data['channels'] = numpy.hstack(
        [numpy.asarray((data_tmp[7][:numsamples])), numpy.asarray((data_tmp[8][:numsamples]))])
    return data

def parse_header(header):
    """ Parsing the header of a MRD file and saving the results in a config dictionary
    """
    funcname = __name__ + '.parse_header():'
    config = {'mss': {'channels':{}}}
    ind_ship = header.find('Ship    :   ') + len('Ship    :   ')
    ship = header[ind_ship:ind_ship+8]
    ship = ship.rstrip('_')
    config['ship'] = ship

    ind_cruise = header.find('Cruise:') + len('Cruise:')
    cruise = header[ind_cruise+1:ind_cruise+18]
    cruise = cruise.rstrip('_')
    config['cruise'] = cruise

    # hs = header.split('\\r')
    hs = header.splitlines()
    sensor_str = []

    logger.debug('PC-Time line:{}'.format(hs[2]))
    config['date_pc'] = dateparser.parse(hs[2]).isoformat()
    #print('Config date',config['date_pc'])
    #print('HS')
    #print('hs',hs)
    #print('HS done')
    #print('Type',type(header))
    # Loop over all channels
    #for i in range(17,len(hs)-1):
    for i in range(len(hs)):
        #logger.debug('Testing line {:s}'.format(hs[i]))
        hstmp = re.sub('\s+',' ',hs[i]) # replace multiple blanks with one
        hsp = hstmp.split(' ')
        # check if we have a sensor line
        # each sensor should have 12 entries
        #print('hsp',hsp,len(hsp))
        if len(hsp)>=12:
            FLAG_sensor = True
            try:
                devicenum = int(hsp[0].replace(' ', ''))
            except:
                FLAG_sensor = False
            try:
                channelnum = int(hsp[2].replace(' ', ''))
            except:
                FLAG_sensor = False
        else:
            FLAG_sensor = False

        if FLAG_sensor:
            #logger.debug(funcname + 'Found sensor')
            sensor_str.append(hs[i])
            devicename = hsp[1].replace(' ', '')
            if('MSS' in devicename): # Treat the MSS here
                config['mss']['name'] = devicename
                config['mss']['devicenum'] = devicenum
                channelnum = int(hsp[2].replace(' ',''))
                channel = hsp[4].replace(' ','')
                unit = hsp[5].replace(' ','')
                caltype = hsp[3].replace(' ','')
                config['mss']['channels'][channelnum] = {}
                config['mss']['channels'][channelnum]['unit']    = unit
                config['mss']['channels'][channelnum]['name']    = channel
                config['mss']['channels'][channelnum]['caltype'] = caltype
                poly = []
                # There are 6 coefficients for each channel
                for val in hsp[6:12]:
                    poly.append(float(val))

                config['mss']['channels'][channelnum]['coeff'] = numpy.asarray(poly)
                #print(devicename,channelnum,channel)
                #if(hs[i].upper().find('COUNT') >=0):
                #    mss = hs[i].split(' ')[1]

    return config

def raw_to_level0(mss_config, rawdata):
    funcname = 'raw_to_units():'
    logger.debug(funcname)
    rawdatac = rawdata['channels']
    #print('rawdata keys',rawdata.keys())
    try:
        gps = rawdata['gps']
        gps_date = rawdata['date']
        logger.debug('Found gps information')
    except:
        logger.debug('No gps information in data')
        gps = None
        pass

    count_offset = mss_config.offset
    if rawdata['numsamples'] == 0:
        logger.info(funcname + ' No samples found for conversion.')
        return None

    # Create matrix for converted data
    # data = numpy.zeros(numpy.shape(rawdatac)) * numpy.nan
    data = {}
    # Create a xarray
    #
    nsamples = len(rawdatac[:, 0])
    index = np.arange(nsamples)
    level0_dataset = xr.Dataset(
        coords={
            "index": index,
        },
    )

    # Add gps information
    level0_dataset.attrs["longitude"] = gps[0][1]
    level0_dataset.attrs["latitude"] = gps[0][2]
    level0_dataset.attrs["date_gps"] = gps_date[0][1].isoformat()
    # Add the header to the dataset
    #try:
    if True:
        headerstr = rawdata['header']
        header = parse_header(headerstr)
        for k in header.keys():
            if 'mss' not in k: # Ignore the mss part
                level0_dataset.attrs["header_" + k] = str(header[k])
    #except:
    #    logger.debug('Could not add header', exc_info=True)

    tstr = datetime.datetime.now().isoformat()

    level0_dataset.attrs["history"] = 'Created with {} (v{}) on the {}'.format(logger.name,version,tstr)

    for s in mss_config.sensors:
        logger.debug('Converting {}'.format(s))
        sensor = mss_config.sensors[s]
        i = sensor.channel - 1 # Channels are saved in index notation starting with 1
        caltype = sensor.calibration_type
        coeff = sensor.coefficients
        unit = sensor.unit
        channelname = sensor.name
        offset = mss_config.offset
        data_tmp = sensor.raw_to_units(rawdatac[:, i], offset)
        data[channelname] = data_tmp
        level0_dataset[channelname] = (["index"], data_tmp)
        level0_dataset[channelname].attrs["units"] = unit
        if 'SHE' in channelname:
            level0_dataset[channelname].attrs["sensitivity"] = sensor.sensitivity

    # print('Config',self.config)
    # Calculate a time [s]
    # TODO: Use the COUNT variable, this is a stupid counting without checking for missing data
    #count_offset_new = len(data['COUNT']) + count_offset
    #data['COUNT_TOTAL'] = np.arange(count_offset, count_offset_new)
    #time_offset = self.config['time_offset_unix']
    #data['t'] = data['COUNT_TOTAL'] / config['fs'] + time_offset
    #return data
    return level0_dataset


def level0_to_level1(mss_config, level0, pspd_rel=None):
    funcname = __name__ + '.level0_to_level1():'
    logger.debug(funcname)
    logger.debug('Pressure sensor:{}'.format(mss_config.pressure_sensorname))
    logger.debug('Keys of level0 data:{}'.format(level0.keys()))
    if True:
        # Loop over all shear sensors and calculate shear
        shearsensors = []
        shear = []
        for k in level0.keys():
            try:
                numshear = int(k[3])
            except:
                numshear = None

            if (('SHE' in k) and (numshear is not None)):
                shearsensors.append(k)

        n_shear = range(len(shearsensors))
        index = level0['index']
        nsamples = len(level0['index'])

        level1_dataset = xr.Dataset(
            coords={
                "n_shear": n_shear,
                "index": index,
            },
        )
        # Count the number of samples and divide by sampling frequency, to get a time vector
        time_count = np.cumsum(index) / mss_config.sampling_freq
        print('time count',time_count)
        print('Time count',np.shape(time_count),np.shape(index))
        channelname = 'time_count'
        level1_dataset[channelname] = time_count
        level1_dataset[channelname].attrs["units"] = 's'
        level1_dataset[channelname].attrs["description"] = 'count of samples divided by sampling frequency'

        # Create a time vector, TODO: This has to be improved alot!
        if False: # Calculate time
            dt = 1 / mss_config.sampling_freq
            t = np.arange(0, nsamples) * dt
            Dt = t[-1]
            if False:
                level1['TIME'] = t
                tstr = self.meta['date'].strftime('seconds since %Y-%m-%d %H:%M:%S +0:00') # TODO allow different time zones
                level1_units['TIME'] = tstr
            else:
                tstr = self.meta['date'].strftime(
                    'seconds since %Y-%m-%d %H:%M:%S +0:00')  # TODO allow different time zones
                level1_units['TIME'] = tstr
                level1['TIME'] = level0['t']


        # Derive salinity, conservative temperature and density
        cond_sensorname = mss_config.sensornames_ctd['cond']
        temp_sensorname = mss_config.sensornames_ctd['temp']
        press_sensorname = mss_config.sensornames_ctd['press']
        cond = level0[cond_sensorname].copy()
        cond[cond<0] = 0
        SP = gsw.SP_from_C(cond, level0[temp_sensorname], level0[press_sensorname])
        level1_dataset['PSAL'] = SP
        level1_dataset['PSAL'].attrs["units"] = '1'
        # Calculating absolute salinity
        print(level0.longitude)
        try:
            lon = level0.longitude
        except:
            lon = 0

        try:
            lat = level0.latitude
        except:
            lat = 0

        SA = gsw.SA_from_SP(SP, level0[press_sensorname],lon,lat)
        level1_dataset['SA'] = SA
        level1_dataset['SA'].attrs["units"] = 'g kg-1'
        # Calculating conservative temperature
        CT = gsw.CT_from_t(SA, level0[temp_sensorname],level0[press_sensorname])
        level1_dataset['CT'] = CT
        level1_dataset['CT'].attrs["units"] = 'degC'
        dens = gsw.rho(SA,CT,level0[press_sensorname])
        level1_dataset['DENS'] = dens
        level1_dataset['DENS'].attrs["units"] = 'kg m-3'
        # Calculating sinking velocity
        try:
            config_pspd_rel = self.config['pspd_rel']
            config_pspd_rel_data = self.config['pspd_rel_data']
        except Exception as e:
            #self.logger.exception(e)
            config_pspd_rel = None

        #print('config', self.config)
        #print('config2', config_pspd_rel,config_pspd_rel_data)
        #if self.config['mss']['pspd_rel'] = 'external'
        if mss_config.pspd_rel_method == 'constant':
            logger.debug(funcname + 'Using constant velocity {:f}'.format(mss_config.pspd_rel_constant_vel))
            vsink = np.zeros(nsamples) + mss_config.pspd_rel_constant_vel
        elif (pspd_rel is None) or (mss_config.pspd_rel_method == 'pressure'):
            logger.debug(funcname + 'Using change of pressure to caluclate velocity')
            vsink = mss_utils.calc_vsink(press=level0[press_sensorname],fs = mss_config.sampling_freq)
        elif mss_config.pspd_rel_method == 'external':
            logger.debug(funcname + 'Using external')
            vsink = pspd_rel
        else:
            self.logger.warning(funcname + ' no method to get velocity past sensor')
            raise ValueError

        level1_dataset['PSPD_REL'] = (["index"], vsink)
        level1_dataset['PSPD_REL'].attrs["units"] = 'm s-1'

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

            if(('SHE' in k) and (numshear is not None)):
                shearsensors.append(k)
                #print('Found shear sensor {:s}, calculating shear'.format(k))
                SH = mss_utils.calc_shear(level0[k], vsink, dens, fs=mss_config.sampling_freq)
                shear.append(SH)

        shear = np.asarray(shear)
        shear[np.isinf(shear)] = np.nan
        level1_dataset['SHEAR'] = (["n_shear","index"], shear)

        return level1_dataset

#
# Legacy ... lets see what will be needed
#

mrd_standard_config = {}

def get_records(config):
    """

    Parameters
    ----------
    config

    Returns
    -------

    """
    records = []
    filenames_orig = config['rawdata']['files']
    if(type(filenames_orig) == list):
        filenames = filenames_orig
    else:
        filenames = glob.glob(filenames_orig)
        logger.info('Expanding files {:s}'.format(filenames_orig))

    filenames.sort()
    for filename in filenames:
        logger.info('Opening file {:s}'.format(filename))
        # Check if only metadata shall be read
        try:
            only_metadata = config['rawdata']['only_metadata']
        except:
            only_metadata = False

        m = mrd(filename,only_metadata=only_metadata)
        records.append(m)

    return records

class mrd():
    """An object to read and process Sea & Sun Technology binary MRD files
    generated by the Microstructure profiler MSS.

    """ 
    def __init__(self, filename=None, only_metadata = False, verbosity=logging.DEBUG, calc_sha1=False, config=None, raw_only=False):
        """

        Args:
            filename:
            process_file:
            only_metadata:
            verbosity:
            calc_sha1:
            userconfig:
            raw_only: read binary data only, no conversion to units
        """
        funcname = __name__ + '.init():'
        self.logger = logging.getLogger('turban.instruments.mss_mrd.mrd')
        self.logger.setLevel(verbosity)
        self.logger.setLevel(logging.DEBUG)

        self.config = {}
        FLAG_interpolate_NAN = True # TODO, add config
        RATIO_BADDATA = 0.1
        FLAG_BADDATA_EXCEEDED = False
        self.filename = filename
        if filename is None:
            self.meta = {}
            self.meta['filename'] = ''
            self.meta['basename'] = ''
            self.meta['cast'] = ''
            self.meta['date'] = datetime.datetime(1970,1,1)
            self.meta['lon'] = np.nan
            self.meta['lat'] = np.nan

            self.logger.debug('No filename given')
            # If the configuration has a key called "header", lets parse it
            try:
                self.config = self.parse_header(config['header'])
                #self.logger.debug('Parsed header: {:s}'.format(str(self.config)))
                self.logger.debug('Parsed header')
            except Exception as e:
                self.logger.exception(e)

            # Add the user defined configuration
            #self.config['postconfig'] = config
            if config is not None:
                self.logger.debug('Updating config')
                self.config.update(config)
                # Gets the time offset of the raw data to have a real time
                self.get_time_offset()

        else:
            self.logger.info(' Opening file: {:s} will full dataset {:s} '.format(filename, str(not (only_metadata))))
            # Get the cast number (a typical mrd file created with SSDA has ????wxyz.mrd with wxyz a 4 digit number)
            basename = os.path.basename(filename)
            try:
                cast = int(os.path.splitext(self.basename)[0][-4:])
            except:
                cast = numpy.nan

            self.file_type = ''
            self.channels = []
            self.data = None
            self.meta = {}
            self.meta['filename'] = filename
            self.meta['basename'] = basename
            self.meta['cast']     = cast
            self.meta['date']     = None
            self.meta['lon']      = np.nan
            self.meta['lat']      = np.nan

            # Opening file for reading and calculating sha1
            try:
                # Calculate a md5 hash
                if(calc_sha1):
                   BLOCKSIZE = 65536
                   hasher = hashlib.sha1()
                   with open(self.filename, 'rb') as afile:
                       buf = afile.read(BLOCKSIZE)
                       while len(buf) > 0:
                          hasher.update(buf)
                          buf = afile.read(BLOCKSIZE)

                   self.sha1 = hasher.hexdigest()
                   afile.close()
                else:
                   self.sha1 = None
            except Exception as e:
                #logger.critical('Could not open file: {:s} ({:s})'.format(self.filename,str(e)))
                self.valid_mrd = False
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), self.filename)
                #return

            # Opening file for reading
            try:
                self.filesize = os.path.getsize(filename)
                self.f        = open(self.filename,'rb')
            except Exception as e:
                #logger.critical('Could not open file:' + self.filename + '( ' + str(e) + ' )')
                self.valid_mrd = False
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), filename)
                return

            # Read the binary data
            rawdata        = self.read_mrd(pos_time_only = only_metadata) # Reading the binary
            self.header = rawdata['header']
            if (len(rawdata['gps']) > 0):
                self.meta['lon'] = rawdata['gps'][0][1]
                self.meta['lat'] = rawdata['gps'][0][2]

            if (len(rawdata['date']) > 0):
                self.meta['date'] = rawdata['date'][0][1]

            self.rawdata        = rawdata
            self.logger.debug('Parsing header')
            self.config         = self.parse_header(self.header)
            self.logger.debug(funcname + 'mrd config: {:s}'.format(str(self.config)))
            # Add the user defined configuration
            # self.config['postconfig'] = config
            if config is not None:
                self.logger.debug('Updating config')
                self.config.update(config)
                # Gets the time offset of the raw data to have a real time
                self.get_time_offset()

            # Gets the time offset of the raw data to have a real time
            try:
                self.config['time_offset_unix']
            except:
                self.config['time_offset_unix'] = 0

            # This needs to be done more smart
            self.config['press_keys'] = ['PRESS','P250','P1000']
            self.meta['ship'] = self.config['ship']
            self.meta['cruise'] = self.config['cruise']
            self.logger.debug('Time offset unix:{}'.format(self.config['time_offset_unix']))
            #print('Config',self.config)
            [self.level0,self.level0_units] = self.raw_to_units(rawdata)
            if raw_only == False:
                self.logger.info('Creating level 1 data')
                [self.level1,self.level1_units] = self.create_level1_data()

    def get_time_offset(self):
        """
        Gets the time offset of the mrd data

        Returns:

        """
        funcname = __name__ + '.get_time_offset():'
        time_offsetunix = 0
        time_offset_format = "%Y.%m.%d %H:%M:%S"
        if 'time_offset' in self.config.keys():
            logger.debug(funcname + 'Using time offset in configuration')
            td_off = datetime.datetime.strptime(self.config['time_offset'],time_offset_format)
            td_off = td_off.replace(tzinfo=timezone('UTC'))
            self.config['time_offset_unix'] = td_off.timestamp()
        else:
            self.config['time_offset_unix'] = 0


    def create_level1_data(self, level0=None, pspd_rel=None, pspd_rel_data=None):
        """
        Creates level1 compatible data.
        see variables names in:
        https://vocab.seadatanet.org/p01-facet-search

        Returns
        -------

        """
        funcname = __name__ + '.create_level1_data():'
        self.logger.debug(funcname)
        if level0 is None:
            level0 = self.level0


        # Find a pressure key
        for k in self.config['press_keys']:
            if k in level0.keys():
                press_key = k


        self.logger.debug('Pressure key:{}'.format(press_key))
        if (level0 is None):
            self.logger.info(funcname + ' Did not find level0 data')
            return [None,None]
        else:
            #
            level1       = {}
            level1_units = {}
            # Create a time vector, TODO: This has to be improved alot!
            nsamples = len(level0['COUNT'])
            dt = 1 / self.config['fs']
            t = np.arange(0, nsamples) * dt
            Dt = t[-1]
            if False:
                level1['TIME'] = t
                tstr = self.meta['date'].strftime('seconds since %Y-%m-%d %H:%M:%S +0:00') # TODO allow different time zones
                level1_units['TIME'] = tstr
            else:
                tstr = self.meta['date'].strftime(
                    'seconds since %Y-%m-%d %H:%M:%S +0:00')  # TODO allow different time zones
                level1_units['TIME'] = tstr
                level1['TIME'] = level0['t']


            self.logger.debug('Keys of level0 data:{}'.format(level0.keys()))
            # Derive salinity, conservative temperature and density
            cond = level0['COND'].copy()
            cond[cond<0] = 0
            SP = gsw.SP_from_C(cond, level0['TEMP'], level0[press_key])
            level1['PSAL'] = SP
            level1_units['PSAL'] = '1'
            # Calculating absolute salinity
            if np.isfinite(self.meta['lon']):
                lon = self.meta['lon']
            else:
                lon = 0

            if np.isfinite(self.meta['lat']):
                lat = self.meta['lat']
            else:
                lat = 0

            SA = gsw.SA_from_SP(SP, level0[press_key],lon,lat)
            level1['SA']       = SA
            level1_units['SA'] = 'g kg-1'
            # Calculating conservative temperature
            CT = gsw.CT_from_t(SA, level0['TEMP'],level0[press_key])
            level1['CT']       = CT
            level1_units['CT'] = 'degC'
            dens = gsw.rho(SA,CT,level0[press_key])
            level1['DENS'] = dens
            level1_units['DENS'] = 'kg m-3'
            # Calculating sinking velocity
            try:
                config_pspd_rel = self.config['pspd_rel']
                config_pspd_rel_data = self.config['pspd_rel_data']
            except Exception as e:
                #self.logger.exception(e)
                config_pspd_rel = None

            #print('config', self.config)
            #print('config2', config_pspd_rel,config_pspd_rel_data)
            #if self.config['mss']['pspd_rel'] = 'external'
            if (pspd_rel is None) and config_pspd_rel == 'external':
                self.logger.debug(funcname + 'Using constant velocity {:f}'.format(config_pspd_rel_data))
                vsink = np.zeros(nsamples) + config_pspd_rel_data
                level1['PSPD_REL'] = vsink
                #vsink = config_pspd_rel_data
            elif (pspd_rel is None) or (pspd_rel.upper() == 'IOW'):
                self.logger.debug(funcname + 'Using IOW velocity')
                vsink = mss_utils.calc_vsink(press=level0[press_key],fs = self.config['fs'])
                level1['PSPD_REL'] = vsink
            elif pspd_rel.upper() == 'DPDT':
                self.logger.debug(funcname + 'Using DPDT')
                vsink = (level0[press_key][-1] - level0[press_key][0])/Dt
                vsink = level0[press_key] * 0 + vsink
                level1['PSPD_REL'] = vsink
            elif pspd_rel.upper() == 'EXTERNAL':
                self.logger.debug(funcname + 'Using external')
                level1['PSPD_REL'] = pspd_rel_data
            else:
                self.logger.warning(funcname + ' no method to get velocity past sensor')


            level1_units['PSPD_REL'] = 'm s-1'
            #
            level1['PRES'] = level0[press_key]
            level1_units['PRES'] = 'dbar'
            #
            level1['TEMP'] = level0['TEMP']
            level1_units['TEMP'] = 'degC'

            # Loop over all shear sensors and calculate shear
            shearsensors = []
            shear = []
            for k in level0.keys():
                try:
                    numshear = int(k[3])
                except:
                    numshear = None

                if(('SHE' in k) and (numshear is not None)):
                    shearsensors.append(k)
                    #print('Found shear sensor {:s}, calculating shear'.format(k))
                    SH = self.calc_shear(level0[k], vsink, dens, fs=self.config['fs'])
                    shear.append(SH)


            shear = np.asarray(shear)
            level1['SHEAR'] = shear.T
            shear[np.isinf(shear)] = np.nan

            return [level1,level1_units]

    def create_level2_data(self, level1=None):
        """
        Level2 data is used to calculate shear, this means that the data needs to be despiked and missing data
        needs to be removed
        Args:
            level1:

        Returns:
            [level2,level2_units]

        """
        funcname = 'create_level2_data():'
        FLAG_interpolate_NAN = True  # TODO, add config
        #FLAG_interpolate_NAN = False  # TODO, add config
        RATIO_BADDATA = 0.1
        FLAG_BADDATA_EXCEEDED = False
        nshear = np.shape(level1['SHEAR'])[1]  # Number of shear sensors
        Ntot = np.shape(level1['SHEAR'])[0]  # Number of shear sensorslen(shear)  # The total length
        level2 = {}
        level2['TIME'] = level1['TIME'].copy()
        level2['SHEAR'] = level1['SHEAR'] * np.nan

        if FLAG_interpolate_NAN:
            t = np.arange(0, Ntot) # Fake time axis
            vars_int = ['PSPD_REL','TEMP','PRES','PSAL']
            for var in vars_int:
                self.logger.debug(funcname + 'Interpolating {:s}'.format(var))
                #print('Hallo', level1)
                #print('Hallo', level1[var])
                nnan = sum(np.isnan(level1[var])) # Count number of nans
                if nnan:
                    logger.debug('Interpolating nan {:d} pspd_rel'.format(nnan))
                    igood = ~np.isnan(level1[var])
                    if sum(igood) > (RATIO_BADDATA * Ntot):
                        level2[var] = np.interp(t, t[igood], level1[var][igood])
                    else:
                        level2[var] = level1[var] * np.nan
                        logger.debug(funcname + 'Will stop processing, bad data exceeds limit')
                        FLAG_BADDATA_EXCEEDED = True
                else:
                    level2[var] = level1[var].copy()

            # Interpolate 2D Matrix, at the moment shear only
            for i in range(nshear):  # Loop over all shear sensors
                nnanshear = sum(np.isnan(level1['SHEAR'][:, i]))
                if nnanshear:
                    print('Interpolating')
                    logger.debug('Interpolating nan {:d} of {:d} SHEAR #{:d}'.format(nnanshear, Ntot, i))
                    igood = ~np.isnan(level1['SHEAR'][:, i])
                    if sum(igood) > (RATIO_BADDATA * Ntot):
                        level2['SHEAR'][:, i] = np.interp(t, t[igood], level1['SHEAR'][igood, i])
                    else:
                        logger.debug(funcname + 'Will stop processing eps, bad data exceeds limit')
                        FLAG_BADDATA_EXCEEDED = True

                    print('shear int',level1['SHEAR'][:, i])
                else:
                    print('All good')
                    level2['SHEAR'][:, i] = level1['SHEAR'][:, i].copy()

        return [level2,None]

    def calc_shear(self,shear,vsink,density,fs):
        """
        Calculates the physical shear by
        * 1 Hz high pass sensor data
        * calculate its time gradient
        * shear = A / (density x Vsink ^ 2)

        Returns
        -------

        """
        dt = 1/fs
        degree = 4
        cutoff_Fs = 1 # 1 Hz
        cutoff = cutoff_Fs / (fs / 2) # non - dim with Nyquist freq.
        [b, a] = scipy.signal.butter(degree, cutoff, 'high')

        shear_hp = scipy.signal.filtfilt(b,a,shear)
        # calculate time gradient of raw shear
        dshdt = mss_utils.gradient(shear_hp, dt)
        # screen for spikes
        dshdt_desp = mss_utils.despike_std(dshdt, 1024, 4)
        #dshdt_desp = dshdt

        vsink_tmp = vsink.copy()
        vsink_tmp[vsink_tmp == 0] = np.nan

        shear      = dshdt_desp * (density **(-1)) * (vsink_tmp**(-2))
        return shear



    def parse_header(self,header):
        """ Parsing the header of the MRD file and saving the results in a config dictionary
        """
        funcname = __name__ + '.parse_header():'
        config = {'mss': {}}
        config['fs'] = 1024 # standard sampling frequency of a MSS
        ind_ship = header.find('Ship    :   ') + len('Ship    :   ')
        ship = header[ind_ship:ind_ship+8]
        ship = ship.rstrip('_')
        config['ship'] = ship

        ind_cruise = header.find('Cruise:') + len('Cruise:')
        cruise = header[ind_cruise+1:ind_cruise+18]
        cruise = cruise.rstrip('_')
        config['cruise'] = cruise

        # hs = header.split('\\r')
        hs = header.splitlines()
        sensor_str = []
        #print('HS')
        #print('hs',hs)
        #print('HS done')
        #print('Type',type(header))
        # Loop over all channels
        #for i in range(17,len(hs)-1):
        for i in range(len(hs)):
            self.logger.debug('Testing line {:s}'.format(hs[i]))
            hstmp = re.sub('\s+',' ',hs[i]) # replace multiple blanks with one
            hsp = hstmp.split(' ')
            # check if we have a sensor line
            # each sensor should have 12 entries
            #print('hsp',hsp,len(hsp))
            if len(hsp)>=12:
                FLAG_sensor = True
                try:
                    devicenum = int(hsp[0].replace(' ', ''))
                except:
                    FLAG_sensor = False
                try:
                    channelnum = int(hsp[2].replace(' ', ''))
                except:
                    FLAG_sensor = False
            else:
                FLAG_sensor = False

            if FLAG_sensor:
                self.logger.debug(funcname + 'Found sensor')
                sensor_str.append(hs[i])
                devicename = hsp[1].replace(' ', '')
                if('MSS' in devicename): # Treat the mss here
                    config['mss']['name'] = devicename
                    # Get the binary offset (data is either calibrated against int16 or uint16)
                    if (devicename == 'MSS038') or (devicename == 'MSS38'):
                        offset = -32768
                    else:
                        offset = 0

                    config['mss']['offset'] = offset
                    config['mss']['devicenum'] = devicenum
                    channelnum = int(hsp[2].replace(' ',''))
                    channel    = hsp[4].replace(' ','')
                    unit       = hsp[5].replace(' ','')
                    caltype    = hsp[3].replace(' ','')
                    config['mss'][channelnum] = {}
                    config['mss'][channelnum]['unit']    = unit
                    config['mss'][channelnum]['name']    = channel
                    config['mss'][channelnum]['caltype'] = caltype
                    poly = []
                    # There are 6 coefficients for each channel
                    for val in hsp[6:12]:
                        poly.append(float(val))

                    config['mss'][channelnum]['coeff'] = numpy.asarray(poly)
                    #print(devicename,channelnum,channel)
                    #if(hs[i].upper().find('COUNT') >=0):
                    #    mss = hs[i].split(' ')[1]


        return config
            
    def get_meta(self):
        """ Returns a dictionary with the metainformation for the record
        """
        info_dict = self.meta
        #info_dict['lon'] = self.lon
        #info_dict['lat'] = self.lat
        #info_dict['date'] = self.date
        #info_dict['file'] = self.filename
        #info_dict['sha1'] = self.sha1
        #info_dict['type'] = 'MRD'
        return info_dict
    

    def read_mrd(self,pos_time_only = False):
        """ Read the binary SST MRD file
        Args:
           pos_time_only: Read as much information to get date and position
        """
        funcname = 'read_mrd():'
        f = self.f

        n = 0
        np = 0 # Number of packets
        end_of_header = 0
        end_of_header_tmp = 0
        IN_HEADER = True
        header    = b''
        HAVE_TIME = False
        HAVE_POS  = False
        
        # The data
        data = {'date':[],'gps':[]}
        data_tmp = {7:[],8:[]}
        nread = 4096
        bind = 0
        btmp = f.read(nread)
        logger.info(funcname + 'Start reading file')
        while True:
            if HAVE_TIME & HAVE_POS & pos_time_only:
                #print('Found time and pos, breaking at',n)
                break

            if(len(btmp) < 17):
                btmp = btmp + f.read(nread)

            if(IN_HEADER):    
                #b = f.read(1)
                b = btmp[0:1]
                btmp = btmp[1:]
                bind += 1
                n += 1        
            else:
                b = btmp[0:17]
                btmp = btmp[17:]
                bind += 17    
                n += 17
                np += 1 # Number of 17 bytes long data packets (packet type + 8 words (little endian)
                
            if len(b) == 0: # Nothing left anymore, break
                break
            else:
                # We are in the data body, lets get our data       
                if(IN_HEADER == False):
                    bword = []
                    for i in range(1,16,2): # Reading 16 bytes and treat them as words
                        bword.append(int.from_bytes(b[i:i+2],byteorder='little'))
                    if(b[0] == 1): # Time packet
                        if(HAVE_TIME == False):
                            year = int(bword[0])
                            year = bword[0]
                            month = bword[1]
                            day = bword[2]
                            hour = bword[4]
                            minute = bword[5]
                            second = bword[6]
                            date = datetime.datetime(year,month,day,hour,minute,second,tzinfo=timezone('UTC'))
                            data['date'].append([np,date])
                            HAVE_TIME = True

                    elif(b[0] == 3): # Position packet
                        if(HAVE_POS == False):
                            # Latitude
                            latint    = int.from_bytes(b[1:3],byteorder='little')
                            latsign   = (latint & 0x8000) >> 15
                            if(latsign == 0):
                                latsign = -1
                            lat       = latint & 0x1FFF
                            latdeg    = (lat - lat%100)/100
                            latmindec = int.from_bytes(b[3:5],byteorder='little')
                            if(abs(latmindec) > 0):                            
                                digits    = int(math.log10(latmindec)) + 1
                            else:
                                digits    = 0
                                
                            latmin    = lat%100 + latmindec / 10**digits
                            latdec    = latsign * (latdeg + latmin/60)
                            # Longitude
                            lonint    = int.from_bytes(b[5:7],byteorder='little')
                            lonsign   = (lonint & 0x8000) >> 15
                            if(lonsign == 0):
                                lonsign = -1                
                            lon       = lonint & 0x1FFF
                            londeg    = (lon - lon%100)/100
                            lonmindec = int.from_bytes(b[7:9],byteorder='little')
                            if(abs(lonmindec) > 0):
                                digits    = int(math.log10(lonmindec)) + 1
                            else:
                                digits    = 0
                                
                            lonmin    = lon%100 + lonmindec / 10**digits
                            londec    = lonsign * (londeg + lonmin/60 )
                            # Daytime
                            tmp = (bword[6]) + (bword[4]&0x000F) * 100000
                            hour = int((tmp - tmp%10000)/10000)
                            tmp2 = tmp%10000
                            minute = int((tmp2 - tmp2%100)/100)
                            second = tmp2%100
                            data['gps'].append([np,londec,latdec,lon,lat,hour,minute,second])
                            HAVE_POS = True

                    elif(b[0] == 7): # Channels 0-7
                        bc = numpy.frombuffer(b[1:], dtype=numpy.uint16)
                        data_tmp[7].append(bc)
                        
                    elif(b[0] == 8): # Channels 8-15
                        bc = numpy.frombuffer(b[1:], dtype=numpy.uint16)
                        data_tmp[8].append(bc)
                        

                else: # We are in the header
                    header += b
                    if(b == b'\x1A'): # Header is filled up with 0x1A until it is dividable by 17
                        end_of_header = n
                        # Check if dividable by 17
                        if((end_of_header)%17 == 0): # Dividable by 17
                            data['valid_mrd'] = True
                            #print('Found a valid header (dividable by 17)')
                            try:
                                data['header'] = header.decode('UTF-8')
                            except:
                                logger.info('UTF-8 did not work, trying to decode using Window charset')
                                data['header'] = header.decode('Windows-1252')
                            IN_HEADER = False

        numsamples = min([len(data_tmp[7]),len(data_tmp[8])])
        data['numsamples'] = numsamples
        data['channels']   = numpy.hstack([numpy.asarray((data_tmp[7][:numsamples])),numpy.asarray((data_tmp[8][:numsamples]))])
        return data


    
    def raw_to_units(self, rawdata, config=None):
        """
        

        Parameters
        ----------
        data : TYPE
            DESCRIPTION.
        config : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        funcname = 'raw_to_units():'
        rawdatac = rawdata['channels']
        if config is None:
            logger.debug(funcname + ': Using internal configuration')
            config = self.config

        try:
            count_offset = rawdata['count_offset']
        except:
            count_offset = 0

        # Add a sampling time if not present
        try:
            config['fs']
        except:
            config['fs'] = 1024

        if rawdata['numsamples'] == 0:
            logger.info(funcname + ' No samples found for conversion.')
            return None

        # Create matrix for converted data
        #data = numpy.zeros(numpy.shape(rawdatac)) * numpy.nan
        data = {}
        data_units = {}
        nchannels = numpy.shape(rawdatac)[1]
        for i in range(nchannels):
            caltype     = config['mss'][i+1]['caltype']
            coeff       = config['mss'][i+1]['coeff']
            unit        = config['mss'][i + 1]['unit']
            channelname = config['mss'][i+1]['name'].upper()
            offset      = config['mss']['offset']
            data_units[channelname] = unit
            # Overwrite with postconfig value
            try:
                offset = config['offset']
                self.logger.info(funcname + ':  Will use userdefined offset of {:d}'.format(offset))
            except:
                pass
            # Treat shear sensors differently, the sensitivities are usually not set in the probe file, but added later
            if channelname.startswith('SHE'):
                logger.debug(funcname + ' Found shear sensor {:s} adding sensitivity to coefficient {:s}'.format(channelname, caltype))
                sens_key = channelname + '_sensitivity'
                try:
                    SP_sens = config[channelname + '_sensitivity']
                    logger.debug('Found user defined sensitivity {:s}, will overwrite value found in the header of the file with {:f}'.format(sens_key,SP_sens))
                    #print('Coeff old',coeff)
                    coeff[0] = -3
                    coeff[1] = 2.94266e-6 / SP_sens
                    # offset
                    offset = -offset # The shear sensors have the negative offset
                    #print('Coeff new', coeff)
                except:
                    self.logger.debug('Could not find user defined sensitivity {:s}, will value defined in the file'.format(sens_key))

            if caltype.upper() == 'N': # Polynom
                self.logger.debug(funcname + 'Converting {:s} with type {:s}'.format(channelname, caltype))
                data_tmp = self.raw_to_units_poly(rawdatac[:,i], coeff,offset=offset)
                #data[:,i] = data_tmp
                data[channelname] = data_tmp
                #input('fsfsfdsfds')
                #print('data conv',data_tmp)
            elif caltype.upper() == 'P': # pressure
                self.logger.debug(funcname + 'Converting {:s} with type {:s}'.format(channelname,caltype))
                data_tmp = self.raw_to_units_pressure(rawdatac[:,i], coeff,offset=offset)
                data[channelname] = data_tmp
                # Check if there is a standard pressure channel
                #print('config',config)
                try:
                    PRESCHANNEL = config['PRESCHANNEL']
                except:
                    PRESCHANNEL = ''

                #print('PRESCHANNEL',PRESCHANNEL,'channelname',channelname)
                if channelname == PRESCHANNEL:
                    self.logger.debug('Found')
                    data['PRES'] = data_tmp
                #input('fsfsfdsfds')
            elif caltype.upper() == 'SHH': # Steinhart/Hart NTC Temperature equation
                logger.debug(funcname + 'Converting {:s} with type {:s}'.format(channelname,caltype))
                data_tmp = self.raw_to_units_shh(rawdatac[:,i], coeff,offset=offset)
                #data[:,i] = data_tmp
                data[channelname] = data_tmp
                #input('fsfsfdsfds')
            elif caltype.upper() == 'NFC': # Turbidity
                logger.debug(funcname + 'Converting {:s} with type {:s}'.format(channelname,caltype))
                data_tmp = self.raw_to_units_nfc(rawdatac[:,i], coeff,offset=offset)
                #data[:,i] = data_tmp
                data[channelname] = data_tmp
                #input('fsfsfdsfds')
            elif caltype.upper() == 'V04': # Oxygen Optode
                logger.debug(funcname + 'Converting {:s} with type {:s}'.format(channelname,caltype))
                data_tmp = self.raw_to_units_oxyoptode(rawdatac[:,i], coeff,offset=offset)
                #data[:,i] = data_tmp
                data[channelname] = data_tmp
                #input('fsfsfdsfds')
            elif caltype.upper() == 'N24': # Oxygen Optode internal Sensor
                logger.debug(funcname + 'Converting {:s} with type {:s}'.format(channelname,caltype))
                data_tmp = self.raw_to_units_poly(rawdatac[:,i], coeff,offset=offset)
                #data[:,i] = data_tmp
                data[channelname] = data_tmp
                #input('fsfsfdsfds')
            else:
                logger.debug(funcname + ' Unknown conversion type ({:s}) for channel {:s} ({:d})'.format(caltype,channelname,i+1))
                #input('dsdfds')

        #print('Config',self.config)
        # Calculate a time [s]
        #TODO: Use the COUNT variable, this is a stupid counting without checking for missing data
        count_offset_new = len(data['COUNT']) + count_offset
        data['COUNT_TOTAL'] = np.arange(count_offset, count_offset_new)
        time_offset = self.config['time_offset_unix']
        data['t'] = data['COUNT_TOTAL'] / config['fs'] + time_offset
        return [data,data_units]
    
    def raw_to_units_poly(self,rawdata,coeff,offset = 0):
        """
        Converts raw 
        
        Parameters
        ----------
        data : TYPE
            DESCRIPTION.
        coeff : TYPE
            DESCRIPTION.
        offset : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        p = numpy.polynomial.Polynomial(coeff)
        data = p(rawdata+offset)
        return data
    
    def raw_to_units_pressure(self,rawdata,coeff,offset = 0):
        """
        
        Parameters
        ----------
        data : TYPE
            DESCRIPTION.
        coeff : TYPE
            DESCRIPTION.
        offset : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        p = numpy.polynomial.Polynomial(coeff[:-1])
        data = p(rawdata + offset) - coeff[-1]
        return data
    
    def raw_to_units_shh(self,rawdata,coeff,offset = 0):
        """
        Steinhart/Hart NTC Polynomial
        
        Parameters
        ----------
        data : TYPE
            DESCRIPTION.
        coeff : TYPE
            DESCRIPTION.
        offset : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        p = numpy.polynomial.Polynomial(coeff)
        data = p(numpy.log(rawdata + offset))
        data = 1/data - 273.15; # Kelvin to degC
        return data
    
    def raw_to_units_nfc(self,rawdata,coeff,offset = 0):
        """
        Convert rawdata turbidity to NFC 
        
        Parameters
        ----------
        data : TYPE
            DESCRIPTION.
        coeff : TYPE
            DESCRIPTION.
        offset : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """        
        p = numpy.polynomial.Polynomial(coeff[:-2])
        data = p(rawdata + offset) * coeff[-1] + coeff[-2]
        return data
    
    def raw_to_units_oxyoptode(self,rawdata,coeff,offset = 0):
        """
        Convert oxygen optode rawdata to physical units
        
        Parameters
        ----------
        data : TYPE
            DESCRIPTION.
        coeff : TYPE
            DESCRIPTION.
        offset : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        p1 = numpy.polynomial.Polynomial(coeff[0:2]) # Convert data to mV
        p2 = numpy.polynomial.Polynomial(coeff[-2:]) # 0 Point correction with B0 and B1
        data_mV = p1(rawdata + offset)
        data = p2(data_mV)
        return data

    def cat_level0_dicts(self,data1,data2):
        """
        Concatenates level0 dictionary data
        Args:
            data1:
            data2:

        Returns:

        """
        data = copy.deepcopy(data1)
        # loop over all keys and do an hstack of the data

        for k in data.keys():
            if k in data2.keys(): # If the data is in the keys, stack it, otherwise just leave it with a copy
                data[k] = np.hstack([data2[k],data1[k]])
            else:
                pass


        return data
    
    def __str__(self):
        """
        String format
        """
        rstr = ""
        rstr += "mrd of " + self.filename
        rstr += " at Lat: " + str(self.meta['lat'])
        rstr += ", Lon: " + str(self.meta['lon'])
        rstr += ", Date: " + datetime.datetime.strftime(self.meta['date'],'%Y-%m-%d %H:%M:%S')
        return rstr                    

