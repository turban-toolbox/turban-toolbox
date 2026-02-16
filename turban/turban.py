import numpy as np

def process_data(rawdata: np.ndarray, use_fast_method: bool = False) -> np.ndarray:
    """
    Scale raw turbulence data by a factor of ten.

    This function applies a linear scaling to the input data. It serves as a
    pre-processing step before further spectral analysis or dissipation
    calculations.

    Parameters
    ----------
    rawdata : np.ndarray
        The input array containing raw sensor measurements (e.g., voltage or
        shear values).
    use_fast_method : bool, optional
        Flag to toggle between a high-precision and a computationally
        optimized processing algorithm. Currently, this parameter is
        reserved for future implementation. Default is False.

    Returns
    -------
    processed_data : np.ndarray
        The scaled data array, same shape and type as `rawdata`.

    """
    # Scaling factor for raw data normalization
    rawdata = rawdata * 10
    return rawdata

