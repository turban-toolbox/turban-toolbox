import xarray as xr
from matplotlib import pyplot as plt
import numpy as np
from turban.utils.util import get_chunking_index
from turban.utils.spectra import remove_vibration_goodman

from tests.filepaths import atomix_benchmark_faroe_fpath


def test_remove_vibration_goodman():
    ds = xr.load_dataset(atomix_benchmark_faroe_fpath, group="L2_cleaned")
    section_number = ds.SECTION_NUMBER.values.astype(int)
    chunk_length = 4096
    chunk_overlap = 2048
    sampfreq = ds.fs_fast

    shear = ds.SHEAR.values
    vib = ds.ACC.values

    reshape_index = get_chunking_index(section_number, (chunk_length, chunk_overlap))

    specarg = dict(
        sampfreq=sampfreq,
        reshape_index=reshape_index,
        segment_length=1024,
        segment_overlap=512,
    )

    # psi_f, freq = spectrum(shear, **specarg) # equivalent to psi_f_cleaned
    psi_f_cleaned, freq, psi_f_uncl = remove_vibration_goodman(shear, vib, **specarg)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.loglog(freq, psi_f_uncl[0, 0].mean(axis=0), label="Uncleaned")
    ax.loglog(freq, psi_f_cleaned[0, 0].mean(axis=0), label="Vibrations removed")
    ax.legend()
    fig.savefig("out/tests/utils/goodman.png")
