from turban.shear import ShearProcessing
import netCDF4
import pylab as pl
from numpy import *


def test_spectra_atomix_baltic():

    atomix_nc_filename = "data/mss/MSS_Baltic.nc"
    print("Opening file: {}".format(atomix_nc_filename))
    nc = netCDF4.Dataset(atomix_nc_filename)

    print("Processing")
    p = ShearProcessing.from_atomix_netcdf(atomix_nc_filename, level=1)

    L1_nc_p = nc.groups["L1_converted"]["PRES"][:]
    L1_nc_sh = nc.groups["L1_converted"]["SHEAR"][:]

    L2_nc_p = L1_nc_p
    L2_nc_sh = nc.groups["L2_cleaned"]["SHEAR"][:]

    L3_nc_k = nc.groups["L3_spectra"].variables["KCYC"][:]
    L3_nc_Pk = nc.groups["L3_spectra"].variables["SH_SPEC"][:]
    L3_nc_p = nc.groups["L3_spectra"].variables["PRES"][:]

    L4_nc_epsi = nc.groups["L4_dissipation"]["EPSI"][:]
    #
    pl.figure(1)
    pl.clf()
    pl.plot(L1_nc_sh[0, :], L1_nc_p)
    pl.plot(p.level1.shear[0, :], L1_nc_p)
    pl.ylim([100, 0])
    pl.title("L1 shear")

    #
    pl.figure(2)
    pl.clf()
    pl.plot(L2_nc_sh[0, :], L2_nc_p)
    pl.plot(p.level2.shear[0, :], L2_nc_p)
    pl.ylim([100, 0])
    pl.title("L2 shear")

    ik = 10
    eps_tur = p.level4.eps[0, ik]
    eps_nc = L4_nc_epsi[0, ik]

    kmin = nc.groups["L4_dissipation"].variables["KMIN"][0, ik]
    kmax = nc.groups["L4_dissipation"].variables["KMAX"][0, ik]

    #
    pl.figure(3)
    pl.clf()
    pnc = pl.plot(L3_nc_k[:, ik], L3_nc_Pk[0, :, ik])
    ptu = pl.plot(p.level3.waveno[ik, :], p.level3.Pk[0, ik, :])
    YL = pl.ylim()
    p3 = pl.plot([kmin] * 2, YL, "-k")
    p4 = pl.plot([kmax] * 2, YL, "-r")
    pl.title(
        "Spectrum in {:.2f} m depth: eps nc 10x{:.2f} eps tu 10x{:.2f}".format(
            L3_nc_p[ik], log10(eps_nc), log10(eps_tur)
        )
    )
    pl.legend([pnc[0], ptu[0], p4[0]], ("netCDF", "turban", "kmax (nc)"))
    pl.gca().set_xscale("log")
    pl.gca().set_yscale("log")
    # pl.plot(p.level2.shear[0,:],L2_nc_p)
    # pl.ylim([100,0])
    # pl.title('L1 shear')

    pl.show()
