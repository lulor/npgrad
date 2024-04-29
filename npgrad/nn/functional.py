from __future__ import annotations

from math import prod as _prod

import numpy as np
import numpy.typing as npt

from ..array import Array
from ..array import asarray_ as _asarray_


def _np_full_like(
    x: npt.NDArray, space_dims: tuple[int, int], fill_value: float
) -> npt.NDArray:
    assert x.ndim >= 2
    return np.full((*x.shape[:-2], *space_dims), fill_value, dtype=x.dtype)


def _np_dilate(x: npt.NDArray, dilation: tuple[int, int]) -> npt.NDArray:
    assert x.ndim >= 2
    d_h, d_w = dilation
    if d_h > 1 or d_w > 1:
        h, w = x.shape[-2:]
        h, w = d_h * (h - 1) + 1, d_w * (w - 1) + 1
        x_dilated = _np_full_like(x, (h, w), 0)
        x_dilated[..., ::d_h, ::d_w] = x
        x = x_dilated
    return x


def _np_pad(
    x: npt.NDArray,
    padding: tuple[int, int],
    fill_value: float = 0,
    copy_input_values: bool = True,
) -> npt.NDArray:
    """
    Pad an ndarray with the specified constant value.

    Parameters
    ----------
    x : ndarray
        The input array to pad.
    padding : tuple of ints
        The padding amount along the last 2 axes. In case of (0, 0), the input array is always returned.
    fill_value : scalar, optional
        The constant value to use to fill the output ndarray (default 0).
    copy_input_values : bool, optional
        If True (default), copy the input array values to the output one (i.e., act as normal padding).
        If False, return an ndarray filled with the specified constant value (i.e., act as full_like).
        In case of padding=(0, 0), this option is ignored and the input array is returned regardless.

    Returns
    -------
    out : ndarray
        The input ndarray if padding=(0, 0), otherwise a new ndarray.
    """
    assert x.ndim >= 2
    p_h, p_w = padding
    if p_h or p_w:
        h, w = x.shape[-2:]
        h, w = h + 2 * p_h, w + 2 * p_w
        x_padded = _np_full_like(x, (h, w), fill_value)
        if copy_input_values:
            max_h, max_w = h - p_h, w - p_w
            x_padded[..., p_h:max_h, p_w:max_w] = x
        x = x_padded
    return x


def _np_trim_padding(
    x: npt.NDArray, padding: tuple[int, int], target_dims: tuple[int, int] | None = None
) -> npt.NDArray:
    assert x.ndim >= 2
    p_h, p_w = padding
    if target_dims:
        h, w = target_dims
        max_h, max_w = h + p_h, w + p_w
    else:
        h, w = x.shape[-2:]
        max_h, max_w = h - p_h, w - p_w
    return x[..., p_h:max_h, p_w:max_w]


def _np_sliding_window(
    x: npt.NDArray,
    window_shape: tuple[int, int],
    stride: tuple[int, int],
    writeable: bool = False,
) -> npt.NDArray:
    assert x.ndim >= 2
    axis = (-2, -1)
    s_h, s_w = stride
    window = np.lib.stride_tricks.sliding_window_view(x, window_shape, axis, writeable=writeable)  # type: ignore
    return window[..., ::s_h, ::s_w, :, :]  # stride


def _np_conv2d(x: npt.NDArray, w: npt.NDArray, stride: tuple[int, int]) -> npt.NDArray:
    # x must be (..., x_h, x_w)
    # w must be (..., w_h, w_w)
    assert x.ndim == w.ndim and x.ndim >= 2

    # sliding window over x -> (..., out_h, out_w, w_h, w_w)
    x = _np_sliding_window(x, w.shape[-2:], stride)  # type: ignore
    w = np.expand_dims(w, (-4, -3))  # -> (..., 1, 1, w_h, w_w)
    out = x * w  # -> (..., out_h, out_w, w_h, w_w)
    out = out.sum((-2, -1))  # sum over window -> (..., out_h, out_w)

    return out


def _tuples(*args: int | tuple[int, int]) -> tuple[tuple[int, int], ...]:
    return tuple(arg if isinstance(arg, tuple) else (arg, arg) for arg in args)


##### nn functions #####


### relu ###


def relu(x: npt.ArrayLike) -> Array:
    x = _asarray_(x)
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
    x: npt.ArrayLike,
    weight: npt.ArrayLike,
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
    dilation: int | tuple[int, int] = 1,
) -> Array:
    x, weight = _asarray_(x), _asarray_(weight)

    ndim = 4
    if x.ndim != ndim or weight.ndim != ndim:
        raise ValueError(
            f"expected {ndim} dimensions for input and weight arrays, but got {x.ndim} and {weight.ndim}"
        )

    stride, padding, dilation = _tuples(stride, padding, dilation)

    # both arrays need to be (n, out_ch, in_ch, h, w)
    x_data = np.expand_dims(x.data, 1)  # -> (n, 1, in_ch, x_h, x_w)
    weight_data = np.expand_dims(weight.data, 0)  # -> (1, out_ch, in_ch, w_h, w_w)

    x_data = _np_pad(x_data, padding)
    weight_data = _np_dilate(weight_data, dilation)

    # trim x
    s_h, s_w = stride
    if s_h > 1 or s_w > 1:
        x_h, x_w = x_data.shape[-2:]
        w_h, w_w = weight_data.shape[-2:]
        x_h -= (x_h - w_h) % s_h
        x_w -= (x_w - w_w) % s_w
        x_data = x_data[..., :x_h, :x_w]  # x_data is a view

    out_data = _np_conv2d(x_data, weight_data, stride).sum(2)  # sum over in_ch

    prevs = tuple(a for a in (x, weight) if a.requires_grad)
    if prevs:
        backward = lambda out: _conv2d_backward(
            out, x, weight, x_data, weight_data, stride, padding, dilation
        )
    else:
        backward = None

    return Array(out_data, requires_grad=bool(prevs), _prevs=prevs, _backward=backward)


def _conv2d_backward(
    out: Array,
    x: Array,
    weight: Array,
    x_data: npt.NDArray,
    weight_data: npt.NDArray,
    stride: tuple[int, int],
    padding: tuple[int, int],
    dilation: tuple[int, int],
) -> None:
    assert out.grad is not None
    out_grad = np.expand_dims(out.grad, 2)  # -> (n, out_ch, 1, out_h, out_w)
    out_grad = _np_dilate(out_grad, stride)

    if x.requires_grad:
        assert x.grad is not None
        w_h, w_w = weight_data.shape[-2:]
        out_grad_ = _np_pad(out_grad, (w_h - 1, w_w - 1))
        weight_data_ = np.flip(np.flip(weight_data, -1), -2)  # rotate by 180
        x_grad = _np_conv2d(out_grad_, weight_data_, (1, 1)).sum(1)
        assert x_grad.shape == x_data.squeeze(1).shape
        x_grad = _np_trim_padding(x_grad, padding, x.shape[-2:])  # type: ignore
        x_h, x_w = x_grad.shape[-2:]  # consider eventual trimming in the forward pass
        x.grad[..., :x_h, :x_w] += x_grad  # ignore trimmed elements

    if weight.requires_grad:
        assert weight.grad is not None
        weight.grad += _np_conv2d(x_data, out_grad, dilation).sum(0)


### max_pool ###


def max_pool2d(
    x: npt.ArrayLike,
    kernel_size: int | tuple[int, int],
    stride: int | tuple[int, int] | None = None,
    padding: int | tuple[int, int] = 0,
) -> Array:
    x = _asarray_(x)

    stride = kernel_size if stride is None else stride
    kernel_size, stride, padding = _tuples(kernel_size, stride, padding)

    x_data = _np_pad(x.data, padding, np.NINF)
    x_data_w = _np_sliding_window(x_data, kernel_size, stride)
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
        x_grad = _np_pad(x.grad, padding, copy_input_values=False)
        x_grad_w = _np_sliding_window(x_grad, kernel_size, stride, writeable=True)
        axis = (-2, -1)
        mask = x_data_w == np.expand_dims(out.data, axis)
        count = np.count_nonzero(mask, axis, keepdims=True)  # type: ignore
        out_grad = np.expand_dims(out.grad, axis) / count
        out_grad = np.broadcast_to(out_grad, x_grad_w.shape)
        np.add.at(x_grad_w, mask, out_grad[mask])
        if any(padding):
            x.grad += _np_trim_padding(x_grad, padding)


### avg_pool ###


def avg_pool2d(
    x: npt.ArrayLike,
    kernel_size: int | tuple[int, int],
    stride: int | tuple[int, int] | None = None,
    padding: int | tuple[int, int] = 0,
) -> Array:
    x = _asarray_(x)

    stride = kernel_size if stride is None else stride
    kernel_size, stride, padding = _tuples(kernel_size, stride, padding)

    x_data = _np_pad(x.data, padding)
    x_data_w = _np_sliding_window(x_data, kernel_size, stride)
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
        x_grad = _np_pad(x.grad, padding, copy_input_values=False)
        x_grad_w = _np_sliding_window(x_grad, kernel_size, stride, writeable=True)
        out_grad = np.expand_dims(out.grad, (-2, -1)) / _prod(kernel_size)
        np.add.at(x_grad_w, slice(None), out_grad)  # type: ignore
        if any(padding):
            x.grad += _np_trim_padding(x_grad, padding)


### softmax / cross_entropy ###


def softmax(x: npt.ArrayLike, axis: int) -> Array:
    x = _asarray_(x)
    if not x.ndim:
        raise ValueError("input must have at least 1 dimension")
    exp_ = np.exp(x - np.amax(x, axis, keepdims=True))
    return exp_ / np.sum(exp_, axis=axis, keepdims=True)


def cross_entropy(x: npt.ArrayLike, target: npt.ArrayLike) -> Array:
    x, target = _asarray_(x), np.asarray(target)

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
