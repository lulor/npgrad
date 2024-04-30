from __future__ import annotations

from math import prod as _prod

import numpy as np
import numpy.typing as npt

from ..array import Array
from ..array import asarray_ as _asarray_
from . import utils

### relu ###


def relu(input: npt.ArrayLike) -> Array:
    x = _asarray_(input)
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


### conv2d ###


def conv2d(
    input: npt.ArrayLike,
    weight: npt.ArrayLike,
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
    dilation: int | tuple[int, int] = 1,
) -> Array:
    x, w = _asarray_(input), _asarray_(weight)

    ndim = 4
    if x.ndim != ndim or w.ndim != ndim:
        raise ValueError(
            f"expected {ndim} dimensions for input and weight arrays, but got {x.ndim} and {w.ndim}"
        )

    stride, padding, dilation = utils.as_tuples(stride, padding, dilation)

    x_data = utils.np_pad(x.data, padding)
    w_data = utils.np_dilate(w.data, dilation)

    # trim x
    s_h, s_w = stride
    if s_h > 1 or s_w > 1:
        x_h, x_w = x_data.shape[-2:]
        w_h, w_w = w_data.shape[-2:]
        x_h -= (x_h - w_h) % s_h
        x_w -= (x_w - w_w) % s_w
        x_data = x_data[..., :x_h, :x_w]  # x_data is a view

    out_data = utils.np_conv2d_v2(x_data, w_data, ("ni", "oi", "no"), stride)

    prevs = tuple(a for a in (x, w) if a.requires_grad)
    if prevs:
        if not x.requires_grad:
            x = w_data = None
        if not w.requires_grad:
            w = x_data = None
        backward = lambda out: _conv2d_backward(
            out, x, w, x_data, w_data, stride, padding, dilation
        )
    else:
        backward = None

    return Array(out_data, requires_grad=bool(prevs), _prevs=prevs, _backward=backward)


def _conv2d_backward(
    out: Array,
    x: Array | None,
    w: Array | None,
    x_data: npt.NDArray | None,
    w_data: npt.NDArray | None,
    stride: tuple[int, int],
    padding: tuple[int, int],
    dilation: tuple[int, int],
) -> None:
    assert out.grad is not None
    out_grad = utils.np_dilate(out.grad, stride)

    if x is not None and x.requires_grad:
        assert x.grad is not None
        assert w_data is not None
        w_h, w_w = w_data.shape[-2:]
        out_grad_ = utils.np_pad(out_grad, (w_h - 1, w_w - 1))  # pad for "full" conv
        w_data = np.flip(np.flip(w_data, -1), -2)  # rotate by 180
        x_grad = utils.np_conv2d_v2(out_grad_, w_data, ("no", "oi", "ni"))
        x_grad = utils.np_trim_padding(x_grad, padding, x.shape[-2:])  # type: ignore
        x_h, x_w = x_grad.shape[-2:]  # consider eventual trimming in the forward pass
        x.grad[..., :x_h, :x_w] += x_grad  # ignore trimmed elements

    if w is not None and w.requires_grad:
        assert w.grad is not None
        assert x_data is not None
        w.grad += utils.np_conv2d_v2(x_data, out_grad, ("ni", "no", "oi"), dilation)


### max_pool ###


def max_pool2d(
    input: npt.ArrayLike,
    kernel_size: int | tuple[int, int],
    stride: int | tuple[int, int] | None = None,
    padding: int | tuple[int, int] = 0,
) -> Array:
    x = _asarray_(input)

    stride = kernel_size if stride is None else stride
    kernel_size, stride, padding = utils.as_tuples(kernel_size, stride, padding)

    x_data = utils.np_pad(x.data, padding, np.NINF)
    x_data_w = utils.np_sliding_window(x_data, kernel_size, stride)
    out_data = x_data_w.max((-2, -1))

    if x.requires_grad:
        prevs = (x,)
        backward = lambda out: _max_pool2d_backward(
            out, x, x_data_w, kernel_size, stride, padding
        )
    else:
        prevs = backward = None

    return Array(
        out_data,
        requires_grad=x.requires_grad,
        _prevs=prevs,
        _backward=backward,
    )


def _max_pool2d_backward(
    out: Array,
    x: Array,
    x_data_w: npt.NDArray,
    kernel_size: tuple[int, int],
    stride: tuple[int, int],
    padding: tuple[int, int],
) -> None:
    assert out.grad is not None
    if x.requires_grad:
        assert x.grad is not None
        # if padding=(0, 0) work on x.grad directly, otherwise work on zero array
        x_grad = utils.np_pad(x.grad, padding, copy_input_values=False)
        x_grad_w = utils.np_sliding_window(x_grad, kernel_size, stride, writeable=True)
        axis = (-2, -1)
        mask = x_data_w == np.expand_dims(out.data, axis)
        count = np.count_nonzero(mask, axis, keepdims=True)  # type: ignore
        out_grad = np.expand_dims(out.grad, axis) / count
        out_grad = np.broadcast_to(out_grad, x_grad_w.shape)
        np.add.at(x_grad_w, mask, out_grad[mask])
        if any(padding):
            x.grad += utils.np_trim_padding(x_grad, padding)


### avg_pool ###


def avg_pool2d(
    input: npt.ArrayLike,
    kernel_size: int | tuple[int, int],
    stride: int | tuple[int, int] | None = None,
    padding: int | tuple[int, int] = 0,
) -> Array:
    x = _asarray_(input)

    stride = kernel_size if stride is None else stride
    kernel_size, stride, padding = utils.as_tuples(kernel_size, stride, padding)

    x_data = utils.np_pad(x.data, padding)
    x_data_w = utils.np_sliding_window(x_data, kernel_size, stride)
    out_data = x_data_w.mean((-2, -1))

    if x.requires_grad:
        prevs = (x,)
        backward = lambda out: _avg_pool2d_backward(
            out, x, kernel_size, stride, padding
        )
    else:
        prevs = backward = None

    return Array(
        out_data,
        requires_grad=x.requires_grad,
        _prevs=prevs,
        _backward=backward,
    )


def _avg_pool2d_backward(
    out: Array,
    x: Array,
    kernel_size: tuple[int, int],
    stride: tuple[int, int],
    padding: tuple[int, int],
) -> None:
    assert out.grad is not None
    if x.requires_grad:
        assert x.grad is not None
        # if padding=(0, 0) work on x.grad directly, otherwise work on zero array
        x_grad = utils.np_pad(x.grad, padding, copy_input_values=False)
        x_grad_w = utils.np_sliding_window(x_grad, kernel_size, stride, writeable=True)
        out_grad = np.expand_dims(out.grad, (-2, -1)) / _prod(kernel_size)
        np.add.at(x_grad_w, slice(None), out_grad)  # type: ignore
        if any(padding):
            x.grad += utils.np_trim_padding(x_grad, padding)


### softmax / cross_entropy ###


def softmax(input: npt.ArrayLike, axis: int) -> Array:
    x = _asarray_(input)
    if not x.ndim:
        raise ValueError("input must have at least 1 dimension")
    exp_ = np.exp(x - np.amax(x, axis, keepdims=True))
    return exp_ / np.sum(exp_, axis=axis, keepdims=True)


def cross_entropy(input: npt.ArrayLike, target: npt.ArrayLike) -> Array:
    x, target = _asarray_(input), np.asarray(target)

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