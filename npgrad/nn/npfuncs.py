from typing import Any

import numpy as np
from numpy.typing import NDArray

from .utils import pair


def _dilate_dims(
    dims: int | tuple[int, int],
    dilation: int | tuple[int, int],
) -> tuple[int, int]:
    h, w = pair(dims)
    d_h, d_w = pair(dilation)
    return d_h * (h - 1) + 1, d_w * (w - 1) + 1


def pad(
    x: NDArray,
    padding: int | tuple[int, int],
    constant_value: float = 0,
) -> NDArray:
    assert x.ndim >= 2
    p_h, p_w = pair(padding)
    if p_h or p_w:
        pad_width = tuple((0,) for _ in range(x.ndim - 2)) + ((p_h,), (p_w,))
        return np.pad(x, pad_width, constant_values=constant_value)
    return x


def crop(x: NDArray, padding: int | tuple[int, int]) -> NDArray:
    assert x.ndim >= 2
    h, w = x.shape[-2:]
    p_h, p_w = pair(padding)
    return x[..., p_h : h - p_h, p_w : w - p_w]


def trim(
    x: NDArray,
    kernel_size: int | tuple[int, int],
    stride: int | tuple[int, int],
    dilation: int | tuple[int, int],
) -> NDArray:
    assert x.ndim >= 2

    s_h, s_w = pair(stride)

    if s_h > 1 or s_w > 1:
        x_h, x_w = x.shape[-2:]
        w_h, w_w = _dilate_dims(kernel_size, dilation)
        x_h -= (x_h - w_h) % s_h
        x_w -= (x_w - w_w) % s_w
        x = x[..., :x_h, :x_w]  # x is a view

    return x


def unfold(
    x: NDArray,
    kernel_size: int | tuple[int, int],
    stride: int | tuple[int, int] = 1,
    dilation: int | tuple[int, int] = 1,
    writeable: bool = False,
) -> NDArray:
    assert x.ndim >= 2

    kernel_size = _dilate_dims(kernel_size, dilation)

    x = np.lib.stride_tricks.sliding_window_view(x, kernel_size, (-2, -1), writeable=writeable)  # type: ignore

    s_h, s_w = pair(stride)
    d_h, d_w = pair(dilation)

    return x[..., ::s_h, ::s_w, ::d_h, ::d_w]


def fold_at(
    x1: NDArray,
    x2: NDArray,
    kernel_size: int | tuple[int, int],
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
    dilation: int | tuple[int, int] = 1,
    indices: Any = None,
) -> None:
    assert x1.ndim == x2.ndim - 2 >= 2

    p_h, p_w = pair(padding)

    if p_h or p_w:
        h, w = x1.shape[-2:]
        x1_pad_shape = x1.shape[:-2] + (h + 2 * p_h, w + 2 * p_w)
        x1_pad = np.zeros_like(x1, shape=x1_pad_shape)
    else:
        x1_pad = x1

    x1_unfolded = unfold(x1_pad, kernel_size, stride, dilation, writeable=True)
    assert (
        x1_unfolded.shape[:-2] == x2.shape[:-2]
    ), f"shape mismatch: {x1_unfolded.shape}, {x2.shape}"

    if indices is None:
        indices = slice(None)
    else:
        x2 = x2[indices]

    np.add.at(x1_unfolded, indices, x2)

    if x1_pad is not x1:
        x1 += crop(x1_pad, padding)


def fold(
    x: NDArray,
    output_size: int | tuple[int, int],
    kernel_size: int | tuple[int, int],
    stride: int | tuple[int, int] = 1,
    dilation: int | tuple[int, int] = 1,
) -> NDArray:
    assert x.ndim >= 4

    out_shape = x.shape[:-4] + pair(output_size)
    out = np.zeros_like(x, shape=out_shape)

    fold_at(out, x, kernel_size, stride, 0, dilation)

    return out
