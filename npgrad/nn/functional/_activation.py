import numpy as np
from numpy.typing import ArrayLike

from npgrad._array import Array, in_array, out_array
from npgrad._grad import is_grad_enabled


def relu(input: ArrayLike) -> Array:
    x = in_array(input)
    out_data = np.maximum(x.data, 0)
    if x.requires_grad and is_grad_enabled():
        prevs = (x,)
        backward = lambda out: _relu_backward(out, x)
    else:
        prevs = backward = None
    return out_array(out_data, prevs, backward)


def _relu_backward(out: Array, x: Array) -> None:
    assert out.grad is not None
    if x.requires_grad:
        assert x.grad is not None
        mask = out.data > 0
        x.grad[mask] += out.grad[mask]


#####


def softmax(input: ArrayLike, axis: int) -> Array:
    x = in_array(input)
    if not x.ndim:
        raise ValueError("input must have at least 1 dimension")
    exp_ = np.exp(x - np.amax(x, axis, keepdims=True))
    return exp_ / np.sum(exp_, axis=axis, keepdims=True)
