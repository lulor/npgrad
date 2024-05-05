import numpy as np
from numpy.typing import ArrayLike

from npgrad._array import Array, asarray_


def cross_entropy(input: ArrayLike, target: ArrayLike) -> Array:
    x, target = asarray_(input), np.asarray(target)

    if x.ndim == 1:
        x = np.expand_dims(x, 0)
        target = np.expand_dims(target, 0)
    elif x.ndim != 2:
        raise ValueError("input must have either 1 or 2 dimensions")

    if target.ndim != x.ndim - 1 or len(target) != len(x):
        raise ValueError(f"x and target shapes mismatch: {x.shape}, {target.shape}")

    # x is (n, c) and target is (n,)

    exp_ = np.exp(x - np.amax(x, axis=1, keepdims=True))
    out = exp_[np.arange(len(target)), target] / np.sum(exp_, axis=1)
    out = -np.log(out)

    return np.squeeze(out)
