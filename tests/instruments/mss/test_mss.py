import numpy as np
import matplotlib.pyplot as plt

from turban.instruments.mss.config import MssDeviceConfig
from turban.instruments.mss.mss_mrd import read_mrd, raw_to_level0, level0_to_level1
from turban.process.shear.api import ShearProcessing, ShearLevel1, ShearConfig
from tests.filepaths import atomix_benchmark_baltic_mrd_fpath


def test_mss():

    mss_conf = MssDeviceConfig.from_mrd(
        filename=atomix_benchmark_baltic_mrd_fpath,
        shear_sensitivities={'SHE1': 3.90e-4, 'SHE2': 4.05e-4},  # sensors 32 and 33 (MSS038)
        offset=0,
    )
    # # Change the pressure sensor manually, because P250 had a cap on and did not measure properly
    # mss_conf.sensornames_ctd["press"] = "P1000"

    with open(atomix_benchmark_baltic_mrd_fpath, "rb") as f:
        data_raw = read_mrd(f)
    data_level0 = raw_to_level0(mss_conf, data_raw)

    print("Attributes", data_level0.attrs)
    data_level1 = level0_to_level1(mss_conf, data_level0)

    #
    shear_config = ShearConfig(
        sampfreq=1024.0,
        segment_length=2048,
        segment_overlap=1024,
        chunk_length=4 * 2048,
        chunk_overlap=1024,
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
    section_number = np.asarray(data_level1["time_count"] * 0, dtype=int)
    section_number[2677:152620] = int(1)
    shear_level1 = ShearLevel1(
        time=np.asarray(data_level1["time_count"]),
        senspeed=np.asarray(data_level1["PSPD_REL"]),
        shear=np.asarray(data_level1["SHEAR"]),
        section_number=section_number,
        cfg=shear_config,
    )

    print(data_level1)

    for v in ["PRESS", "TEMP", "COND"]:
        shear_level1.add_aux_data(data_level1[v].values, v.lower())

    p = ShearProcessing(shear_level1)
    level4 = p.level4.to_xarray()

    fig, ax = plt.subplots()
    ax.plot(level4["eps"][0, :], level4["press_mean"])
    ax.plot(level4["eps"][1, :], level4["press_mean"])
    ax.set_xscale("log")
    ax.invert_yaxis()
    ax.set_ylabel("Pressure [dbar]")
    ax.set_xlabel("Epsilon [W kg-1]")
    ax.legend(["Shear 1", "Shear 2"])
    fig.savefig("out/tests/instruments/mss/SH2_0330.png")
