from tests.fixtures import mss_mrd_filename
import turban
from turban.instruments.mss import mss
from turban.shear import ShearLevel1, ShearProcessing, ShearConfig


def test_read_mrd(mss_mrd_filename):
    mrddata = turban.instruments.mss.mss_mrd.mrd(mss_mrd_filename)
    print(mrddata.level0['PRESS'])


mss_configs = {}
mss_configs['eddy0001.MRD'] = {'filename':'../../data/mss/eddy0001.MRD','shear_sensitivities':[2.28e-4, 2.53e-4],'offset':0} # sensors 5 and 7 (MSS010)
mss_configs['M87_0414.MRD'] = {'filename':'../../data/mss/M87_0414.MRD','shear_sensitivities':[4.39e-4, 4.08e-4],'offset':0} # sensors 70 and 71 (MSS055)

dataset = 'M87_0414.MRD'
mss_conf = mss.MssDeviceConfig().from_mrd(filename = mss_configs[dataset]['filename'], shear_sensitivities=mss_configs[dataset]['shear_sensitivities'], offset=mss_configs[dataset]['offset'])
# Change the pressure sensor manually, because P250 had a cap on and did not measure properly
mss_conf.sensornames_ctd['press'] = 'P1000'
#print('MSS conf')
#print(mss_conf)
filestream = open('../../data/mss/M87_0414.MRD','rb')
data_raw = mss.mss_mrd.read_mrd(filestream)
#print('Data',data_raw.keys())
data_level0 = mss.mss_mrd.raw_to_level0(mss_conf, data_raw)
print('Saving level0 data to netcdf')
data_level0.to_netcdf(dataset + '.nc')
print('Saving level0 data to netcdf done')
#vsink = mss.mss_mrd.mss_utils.calc_vsink(press=data_level0['P250'],fs = mss_config.sampling_freq)
print('Attributes',data_level0.attrs)
data_level1 = mss.mss_mrd.level0_to_level1(mss_conf, data_level0)
print('Saving level1 data to netcdf')
data_level1.to_netcdf(dataset + '_level1.nc')

#
shear_config = ShearConfig(sampling_freq=1024,fft_length= 1024,fft_overlap=512,diss_length=1024,diss_overlap=512)
section_marker = arange(1,len(data_level1['PSPD_REL']))
ShearLevel1(
            time=asarray(data_level1['time_count']),
            pspd=asarray(data_level1['PSPD_REL']),
            shear=asarray(data_level1['SHEAR']),
            section_marker=section_marker,
            cfg=shear_config,
        )

p = ShearProcessing(ShearLevel1, level=1)
