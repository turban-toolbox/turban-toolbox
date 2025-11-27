import numpy as np
from numpy import ndarray
from jaxtyping import Int

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from turban.utils.util import unwrap_base2

def plot_quality_metric(q: Int[ndarray, "*any nshear time"], **kwarg):
    green_red_cmap = LinearSegmentedColormap.from_list("GreenRedCmap", ["green", "red"])

    flag_dict = unwrap_base2(q, **kwarg)
    nshear = q.shape[-2]
    q = np.stack(list(flag_dict.values()), axis=0)
    flag = list(flag_dict.keys())

    fig, axs = plt.subplots(nrows=nshear, figsize=(10, 3 + nshear * 2))
    for k, ax in enumerate(axs):
        ax.imshow(q[..., k, :], cmap=green_red_cmap)
        ax.set_xlabel("Time")
        ax.set_yticks(range(len(flag)), flag)
        ax.grid()
        ax.set_title(f"Quality metric for shear sensor #{k+1}")

    return fig