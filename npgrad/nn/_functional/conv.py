from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from ..._array import Array, asarray_
from . import _np_utils as npu

_EINSUM_OPTIM = "optimal"


def conv2d(
    input: ArrayLike,
    weight: ArrayLike,
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
    dilation: int | tuple[int, int] = 1,
) -> Array:
    x, w = asarray_(input), asarray_(weight)

    ndim = 4
    if x.ndim != ndim or w.ndim != ndim:
        raise ValueError(
            f"expected {ndim} dimensions for input and weight arrays, but got {x.ndim} and {w.ndim}"
        )

    x_data = npu.pad(x.data, padding)
    x_data = npu.unfold(x_data, w.shape[-2:], stride, dilation)  # type: ignore

    out_data = np.einsum("nixyhw,oihw->noxy", x_data, w.data, optimize=_EINSUM_OPTIM)

    prevs = tuple(a for a in (x, w) if a.requires_grad)

    if prevs:
        backward = lambda out: _conv2d_backward(out, x, w, stride, padding, dilation)
    else:
        backward = None

    return Array(out_data, requires_grad=bool(prevs), _prevs=prevs, _backward=backward)


def _conv2d_backward(
    out: Array,
    x: Array,
    w: Array,
    stride: int | tuple[int, int],
    padding: int | tuple[int, int],
    dilation: int | tuple[int, int],
) -> None:
    assert out.grad is not None

    if x.requires_grad:
        assert x.grad is not None
        x_grad = np.einsum(
            "noxy,oihw->nixyhw", out.grad, w.data, optimize=_EINSUM_OPTIM
        )
        npu.fold_at(x.grad, x_grad, w.shape[-2:], stride, padding, dilation)  # type: ignore

    if w.requires_grad:
        assert w.grad is not None
        x_data = npu.pad(x.data, padding)
        x_data = npu.trim(x_data, w.shape[-2:], stride, dilation)  # type: ignore
        x_data = npu.unfold(x_data, out.shape[-2:], stride=dilation, dilation=stride)  # type: ignore
        w.grad += np.einsum(
            "nihwxy,noxy->oihw", x_data, out.grad, optimize=_EINSUM_OPTIM
        )
