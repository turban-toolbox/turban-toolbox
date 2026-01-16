from typing import Iterable

import numpy as np
from numpy import ndarray
import matplotlib as mpl
import matplotlib.pyplot as plt
from jaxtyping import Int, Shaped
from matplotlib.colors import LinearSegmentedColormap

from turban.utils.util import unwrap_base2


def plot_section_numbers(
    axs: Iterable[mpl.axes.Axes],
    time: Shaped[np.ndarray, "n"],
    section_num: Int[np.ndarray, "n"],
):
    # Plot section_number as solid bars where non-zero
    non_zero_mask = section_num != 0

    # Find continuous sections
    changes = np.diff(np.concatenate(([False], non_zero_mask, [False])).astype(int))
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]

    for ax in axs:
        for start, end in zip(starts, ends):
            ax.axvspan(
                time[start],
                time[end - 1],
                alpha=0.1,
                color="green",
            )


def plot_quality_metric(
    ax,
    time: Shaped[ndarray, "*any time"],
    q: Int[ndarray, "*any time"],
    **kwarg,
):
    green_red_cmap = LinearSegmentedColormap.from_list("GreenRedCmap", ["green", "red"])

    flag_dict = unwrap_base2(q, **kwarg)
    q = np.stack(list(flag_dict.values()), axis=0)
    flag = list(flag_dict.keys())

    ax.pcolormesh(time, range(len(flag)), q, cmap=green_red_cmap)
    ax.set_yticks(range(len(flag)), flag)
