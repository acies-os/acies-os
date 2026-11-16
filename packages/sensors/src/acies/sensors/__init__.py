import logging

import numpy as np
import numpy.typing as npt

logging.getLogger(__name__).addHandler(logging.NullHandler())

NORMALIZATION_CONSTANT = 1e9


def signal_energy(samples: npt.NDArray[np.number] | bytes, dtype: str | None = None) -> float:
    """Compute signal energy (sum of squares) as a float64 scalar.

    Args:
        samples: numpy array or raw bytes. If bytes, ``dtype`` must be
            provided to decode via ``np.frombuffer``.
        dtype: numpy dtype string (e.g. 'int32'). Required when
            ``samples`` is bytes.
    """
    if isinstance(samples, bytes):
        arr = np.frombuffer(samples, dtype=np.dtype(dtype))
    else:
        arr = samples
    return float(np.sum(np.square(arr, dtype=np.float64))) / NORMALIZATION_CONSTANT
