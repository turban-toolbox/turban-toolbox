from pathlib import Path

top_level = Path(__file__).resolve().parent.parent


atomix_benchmark_baltic_fpath = str(top_level / "data/process/shear/MSS_Baltic.nc")
atomix_benchmark_faroe_fpath = str(
    top_level / "data/process/shear/VMP2000_FaroeBankChannel.nc"
)
atomix_benchmark_baltic_mrd_fpath = str(top_level / "data/instruments/mss/SH2_0330.MRD")
mss_mrd_fpath = str(top_level / "data/instruments/mss/Nien0020.MRD")

mss_probeconf_json_fpath = str(
    top_level / "data/instruments/mss/probeconf_mss053_2024.json"
)
mss_utemp_mrd_fpath = str(top_level / "data/instruments/mss/probeconf_mss053_2024.json")
