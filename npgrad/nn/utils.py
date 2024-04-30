from __future__ import annotations

from contextlib import contextmanager
from time import perf_counter

import numpy as np
import numpy.typing as npt


@contextmanager
def timed(s: str | None = None):
    start = perf_counter()
    yield
    duration = perf_counter() - start
    prefix = f"{s} " if s else ""
    print(f"{prefix}time: {duration}")


def tuples(*args: int | tuple[int, int]) -> tuple[tuple[int, int], ...]:
    return tuple(arg if isinstance(arg, tuple) else (arg, arg) for arg in args)


def np_full_like(
    x: npt.NDArray, space_dims: tuple[int, int], fill_value: float
) -> npt.NDArray:
    assert x.ndim >= 2
    return np.full((*x.shape[:-2], *space_dims), fill_value, dtype=x.dtype)


def np_dilate(x: npt.NDArray, dilation: tuple[int, int]) -> npt.NDArray:
    assert x.ndim >= 2
    d_h, d_w = dilation
    if d_h > 1 or d_w > 1:
        h, w = x.shape[-2:]
        h, w = d_h * (h - 1) + 1, d_w * (w - 1) + 1
        x_dilated = np_full_like(x, (h, w), 0)
        x_dilated[..., ::d_h, ::d_w] = x
        x = x_dilated
    return x


def np_pad(
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
        x_padded = np_full_like(x, (h, w), fill_value)
        if copy_input_values:
            max_h, max_w = h - p_h, w - p_w
            x_padded[..., p_h:max_h, p_w:max_w] = x
        x = x_padded
    return x


def np_trim_padding(
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


def np_sliding_window(
    x: npt.NDArray,
    window_shape: tuple[int, int],
    stride: tuple[int, int] | None = None,
    writeable: bool = False,
) -> npt.NDArray:
    assert x.ndim >= 2
    axis = (-2, -1)
    window = np.lib.stride_tricks.sliding_window_view(x, window_shape, axis, writeable=writeable)  # type: ignore
    if stride is not None:
        s_h, s_w = stride
        return window[..., ::s_h, ::s_w, :, :]
    return window


def np_conv2d(
    x: npt.NDArray, w: npt.NDArray, stride: tuple[int, int] | None = None
) -> npt.NDArray:
    # x must be (..., x_h, x_w)
    # w must be (..., w_h, w_w)
    assert x.ndim == w.ndim >= 2

    # sliding window over x -> (..., out_h, out_w, w_h, w_w)
    x = np_sliding_window(x, w.shape[-2:], stride)  # type: ignore
    w = np.expand_dims(w, (-4, -3))  # -> (..., 1, 1, w_h, w_w)
    # (..., out_h, out_w, w_h, w_w) -> (..., out_h, out_w)
    out = np.sum(x * w, (-2, -1))

    return out


def np_conv2d_v2(
    x: npt.NDArray,
    w: npt.NDArray,
    labels: tuple[str, str, str],
    stride: tuple[int, int] | None = None,
) -> npt.NDArray:
    assert x.ndim == w.ndim >= 2

    x_, w_, out_ = labels
    assert len(x_) == len(w_) == len(out_)

    x = np_sliding_window(x, w.shape[-2:], stride)  # type: ignore
    subscripts = f"{x_}xyhw,{w_}hw->{out_}xy"
    out = np.einsum(subscripts, x, w, optimize="optimal")

    return out
