import xarray as xr
from turban.process.shear.api import ShearProcessing, ShearLevel1, ShearConfig
import matplotlib.pyplot as plt

from tests.filepaths import atomix_benchmark_faroe_fpath

def test_compare_turban_to_atomix_faroe_example():

    cfg = ShearConfig.from_atomix_netcdf(atomix_benchmark_faroe_fpath)

    ds1 = xr.load_dataset(atomix_benchmark_faroe_fpath, group="L1_converted")
    ds2 = xr.load_dataset(atomix_benchmark_faroe_fpath, group="L2_cleaned")
    ds4 = xr.load_dataset(atomix_benchmark_faroe_fpath, group="L4_dissipation")

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
