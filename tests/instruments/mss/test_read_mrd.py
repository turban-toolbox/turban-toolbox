from tests.fixtures import mss_mrd_filename
import turban
from turban.instruments.mss import mss
from turban.process.shear.api import ShearProcessing, ShearLevel1, ShearConfig
import numpy as np
import matplotlib.pyplot as plt

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
shear_config = ShearConfig(
    sampling_freq=1024.0,
    segment_length=2048,
    segment_overlap=1024,
    diss_length=5120,
    diss_overlap=2560,
    freq_cutoff_antialias=999.0,
    freq_cutoff_corrupt=999.0,
    freq_highpass=0.15,
    spatial_response_wavenum=50.0,
    waveno_cutoff_spatial_corr=999.0,
    spike_threshold=8.0,
    max_tries=10,
    spike_replace_before=512,
    spike_replace_after=512,
    spike_include_before=10,
    spike_include_after=20,
    cutoff_freq_lp=0.5,
)
section_number= np.asarray(data_level1['time_count'] * 0,dtype=int)
section_number[2677:172620] = int(1)
shear_level1 = ShearLevel1(
            time=np.asarray(data_level1['time_count']),
            pspd=np.asarray(data_level1['PSPD_REL']),
            shear=np.asarray(data_level1['SHEAR']),
            section_number=section_number,
            cfg=shear_config,
        )


#aux_vars = ["time", "press", "temp", "cond"]
aux_vars = ["time", "P1000", "Temp", "Cond"]
data_aux = {
    "time": (
        ["time"],
        np.asarray(data_level1["time_count"]),
        {"mean": "time_slow"},
    ),
    "press": (
        ["time"],
        np.asarray(data_level1["P1000"]),
        {"mean": "press"},
    ),
    "temp": (
        ["time"],
        np.asarray(data_level1["Temp"]),
        {"mean": "temp"},
    ),
    "cond": (
        ["time"],
        np.asarray(data_level1["Cond"]),
        {"mean": "cond"},
    ),
}
coords_aux = ["time", "time_slow"]
p = ShearProcessing(shear_level1, level=1, data_aux=data_aux, coords_aux=coords_aux)
level4 = p.level4.to_xarray()
aux_data_processed, _ = p.aux.to_xarray()

# Plot the results
plt.figure(1)
plt.clf()
plt.plot(np.log10(level4['eps'][0,:]),aux_data_processed['press'])
plt.plot(np.log10(level4['eps'][1,:]),aux_data_processed['press'])
plt.ylim(100,0)
plt.ylabel('Pressure [dbar]')
plt.xlabel('Epsilon [W kg-1]')
