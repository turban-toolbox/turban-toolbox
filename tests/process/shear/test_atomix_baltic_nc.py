import matplotlib.pyplot as plt
import netCDF4
import numpy as np

from turban.process.shear.api import ShearProcessing
from tests.filepaths import atomix_benchmark_baltic_fpath

def test_spectra_atomix_baltic():

    print("Opening file: {}".format(atomix_benchmark_baltic_fpath))
    nc = netCDF4.Dataset(atomix_benchmark_baltic_fpath)

    print("Processing")
    p = ShearProcessing.from_atomix_netcdf(atomix_benchmark_baltic_fpath, level=1)

    L1_nc_p = nc.groups["L1_converted"]["PRES"][:]
    L1_nc_sh = nc.groups["L1_converted"]["SHEAR"][:]

    L2_nc_p = L1_nc_p
    L2_nc_sh = nc.groups["L2_cleaned"]["SHEAR"][:]

    L3_nc_k = nc.groups["L3_spectra"].variables["KCYC"][:]
    L3_nc_psi_k_sh = nc.groups["L3_spectra"].variables["SH_SPEC"][:]
    L3_nc_p = nc.groups["L3_spectra"].variables["PRES"][:]

    L4_nc_epsi = nc.groups["L4_dissipation"]["EPSI"][:]
    #
    fig, ax = plt.subplots()
    ax.plot(L1_nc_sh[0, :], L1_nc_p)
    ax.plot(p.level1.shear[0, :], L1_nc_p)
    ax.set_ylim([100, 0])
    ax.set_title("L1 shear")
    fig.savefig("out/tests/baltic-level1.png")

    #
    fig, ax = plt.subplots()
    ax.plot(L2_nc_sh[0, :], L2_nc_p)
    ax.plot(p.level2.shear[0, :], L2_nc_p)
    ax.set_ylim([100, 0])
    ax.set_title("L2 shear")
    fig.savefig("out/tests/baltic-level2.png")

    ik = 10
    eps_tur = p.level4.eps[0, ik]
    eps_nc = L4_nc_epsi[0, ik]

    kmin = nc.groups["L4_dissipation"].variables["KMIN"][0, ik]
    kmax = nc.groups["L4_dissipation"].variables["KMAX"][0, ik]

    fig, ax = plt.subplots()
    pnc = ax.plot(L3_nc_k[:, ik], L3_nc_psi_k_sh[0, :, ik])
    ptu = ax.plot(p.level3.waveno[ik, :], p.level3.psi_k_sh[0, ik, :])
    YL = ax.get_ylim()
    p3 = ax.plot([kmin] * 2, YL, "-k")
    p4 = ax.plot([kmax] * 2, YL, "-r")
    ax.set_title(
        "Spectrum in {:.2f} m depth: eps nc 10x{:.2f} eps tu 10x{:.2f}".format(
            L3_nc_p[ik], np.log10(eps_nc), np.log10(eps_tur)
        )
    )
    ax.legend([pnc[0], ptu[0], p4[0]], ("netCDF", "turban", "kmax (nc)"))
    ax.set_xscale("log")
    ax.set_yscale("log")
    fig.savefig("out/tests/baltic-spectra.png")
