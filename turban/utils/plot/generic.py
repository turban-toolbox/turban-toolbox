from typing import Iterable

import numpy as np
import matplotlib as mpl
from jaxtyping import Int, Shaped


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
