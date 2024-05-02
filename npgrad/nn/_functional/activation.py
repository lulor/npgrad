from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from ..._array import Array, asarray_

##### relu #####


def relu(input: ArrayLike) -> Array:
    x = asarray_(input)
    if x.requires_grad:
        prevs = (x,)
        backward = lambda out: _relu_backward(out, x)
    else:
        prevs = backward = None
    return Array(
        np.maximum(x.data, 0),
        requires_grad=x.requires_grad,
        _prevs=prevs,
        _backward=backward,
    )


def _relu_backward(out: Array, x: Array) -> None:
    assert out.grad is not None
    if x.requires_grad:
        assert x.grad is not None
        mask = out.data > 0
        x.grad[mask] += out.grad[mask]


##### softmax #####


def softmax(input: ArrayLike, axis: int) -> Array:
    x = asarray_(input)
    if not x.ndim:
        raise ValueError("input must have at least 1 dimension")
    exp_ = np.exp(x - np.amax(x, axis, keepdims=True))
    return exp_ / np.sum(exp_, axis=axis, keepdims=True)
