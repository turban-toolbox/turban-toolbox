import numpy as np
import xarray as xr
from turban.process.shear.api import ShearProcessing, ShearLevel1, ShearConfig
import matplotlib.pyplot as plt


def test_compare_turban_to_atomix_faroe_example():
    atomix_nc_filename = "./data/shear/VMP2000_FaroeBankChannel.nc"

    cfg = ShearConfig(
        sampfreq=512.0,
        segment_length=2048,
        segment_overlap=1024,
        chunk_length=4096,  # 8s*512
        chunk_overlap=2048,  # 4s*512
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

    ds1 = xr.load_dataset(atomix_nc_filename, group="L1_converted")
    ds2 = xr.load_dataset(atomix_nc_filename, group="L2_cleaned")
    ds3 = xr.load_dataset(atomix_nc_filename, group="L3_spectra")
    ds4 = xr.load_dataset(atomix_nc_filename, group="L4_dissipation")

    time = ds1.TIME.values  # .astype("datetime64[ns]").astype(
    #     np.float64
    # )  # time in seconds since epoch
    level1 = ShearLevel1(
        time=time,
        senspeed=ds2.PSPD_REL.values,
        cfg=cfg,
        shear=ds1.SHEAR.values,
        section_number=ds2.SECTION_NUMBER.values.astype(int),
    )

    p = ShearProcessing(level1)

    l4 = p.level4.to_xarray()

    fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    ds4.set_coords("TIME")
    ds4 = ds4.swap_dims({"TIME_SPECTRA": "TIME"})

    a = 0
    ds4.EPSI_FINAL.plot(
        ax=axs[a],
        marker="x",
        color="#2E2626",
        linewidth=3,
        label="EPSI_FINAL benchmark",
    )
    ds4.EPSI.isel(N_SHEAR_SENSORS=0).plot(
        ax=axs[a], x="TIME", marker="x", label="EPSI S1 benchmark"
    )
    ds4.EPSI.isel(N_SHEAR_SENSORS=1).plot(
        ax=axs[a], marker="x", label="EPSI S2 benchmark"
    )
    l4.eps.isel(nshear=0).plot(ax=axs[a], marker="x", label="EPSI S1 TURBAN")
    l4.eps.isel(nshear=1).plot(ax=axs[a], marker="x", label="EPSI S2 TURBAN")
    axs[a].set_yscale("log")
    axs[a].legend()

    a += 1
    r1 = l4.eps.isel(nshear=0).values / ds4.EPSI.isel(N_SHEAR_SENSORS=0).values
    r2 = l4.eps.isel(nshear=1).values / ds4.EPSI.isel(N_SHEAR_SENSORS=1).values

    axs[a].plot(l4.time.values, r1, marker="x", label="EPSI S1 ratio")
    axs[a].plot(l4.time.values, r2, marker="x", label="EPSI S2 ratio")
    axs[a].axhline(1.0, color="k", linestyle="--")
    axs[a].set_ylabel("Ratio TURBAN/benchmark")
    axs[a].legend()
    axs[a].set_yscale("log")

    a += 1
    ds1.PRES.plot(ax=axs[a])

    fig.savefig("out/tests/process/shear/faroe-benchmark.png")
