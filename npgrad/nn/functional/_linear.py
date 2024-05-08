from numpy.typing import ArrayLike

from npgrad._array import Array, asarray_


def linear(
    input: ArrayLike,
    weight: ArrayLike,
    bias: ArrayLike | None = None,
) -> Array:
    x, w = asarray_(input), asarray_(weight)
    b = bias if bias is None else asarray_(bias)

    if x.ndim not in (1, 2):
        raise ValueError(f"input must have 1 or 2 dims (got {x.ndim})")
    if w.ndim not in (1, 2):
        raise ValueError(f"weight must have 1 or 2 dims (got {w.ndim})")
    if b is not None:
        if b.ndim != w.ndim - 1:
            raise ValueError(f"weight and bias shape mismatch: {w.shape}, {b.shape}")

    out = x @ w.T

    return out if b is None else out + b
