import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

import npgrad.nn.functional._np_utils as npu
from npgrad._array import Array, in_array, out_array
from npgrad._grad import is_grad_enabled
from npgrad.nn._utils import pair


def max_pool2d(
    input: ArrayLike,
    kernel_size: int | tuple[int, int],
    stride: int | tuple[int, int] | None = None,
    padding: int | tuple[int, int] = 0,
    dilation: int | tuple[int, int] = 1,
) -> Array:
    if stride is None:
        stride = kernel_size

    x = in_array(input)
    x_data = npu.pad(x.data, padding, np.NINF)
    x_data = npu.unfold(x_data, kernel_size, stride, dilation)

    axis = (-2, -1)
    out_data = x_data.max(axis)

    if x.requires_grad and is_grad_enabled():
        prevs = (x,)
        mask = x_data == np.expand_dims(out_data, axis)
        backward = lambda out: _max_pool2d_backward(
            out, x, mask, kernel_size, stride, padding, dilation
        )
    else:
        prevs = backward = None

    return out_array(out_data, prevs, backward)


def _max_pool2d_backward(
    out: Array,
    x: Array,
    mask: NDArray,
    kernel_size: int | tuple[int, int],
    stride: int | tuple[int, int],
    padding: int | tuple[int, int],
    dilation: int | tuple[int, int],
) -> None:
    assert out.grad is not None
    if x.requires_grad:
        assert x.grad is not None
        axis = (-2, -1)
        count = np.count_nonzero(mask, axis, keepdims=True)  # type: ignore
        out_grad = np.expand_dims(out.grad, axis) / count
        out_grad = np.broadcast_to(out_grad, mask.shape)
        npu.fold_at(
            x.grad, out_grad, kernel_size, stride, padding, dilation, indices=mask
        )


#####


def avg_pool2d(
    input: ArrayLike,
    kernel_size: int | tuple[int, int],
    stride: int | tuple[int, int] | None = None,
    padding: int | tuple[int, int] = 0,
) -> Array:
    if stride is None:
        stride = kernel_size

    x = in_array(input)
    x_data = npu.pad(x.data, padding)
    x_data = npu.unfold(x_data, kernel_size, stride)

    out_data = x_data.mean((-2, -1))

    if x.requires_grad and is_grad_enabled():
        prevs = (x,)
        backward = lambda out: _avg_pool2d_backward(
            out, x, kernel_size, stride, padding
        )
    else:
        prevs = backward = None

    return out_array(out_data, prevs, backward)


def _avg_pool2d_backward(
    out: Array,
    x: Array,
    kernel_size: int | tuple[int, int],
    stride: int | tuple[int, int],
    padding: int | tuple[int, int],
) -> None:
    assert out.grad is not None
    if x.requires_grad:
        assert x.grad is not None
        out_grad = np.expand_dims(out.grad, (-2, -1)) / math.prod(pair(kernel_size))
        npu.fold_at(x.grad, out_grad, kernel_size, stride, padding)
