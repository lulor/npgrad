import numpy as np
from numpy.typing import ArrayLike

import npgrad.nn.functional._np_utils as npu
from npgrad._array import Array, in_array, out_array
from npgrad._grad import is_grad_enabled

_EINSUM_OPTIM = True


def conv2d(
    input: ArrayLike,
    weight: ArrayLike,
    bias: ArrayLike | None = None,
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
    dilation: int | tuple[int, int] = 1,
) -> Array:
    x, w = in_array(input), in_array(weight)
    b = bias if bias is None else in_array(bias)

    if not (x.ndim == w.ndim == 4):
        raise ValueError(
            f"expected 4 dimensions for input and weight, but got {x.ndim} and {w.ndim}"
        )
    if b is not None:
        if b.ndim != 1:
            raise ValueError(f"expected 1 dim for bias, but got {b.ndim}")
        if b.shape[0] != w.shape[0]:
            raise ValueError(f"weight and bias shape mismatch: {w.shape}, {b.shape}")

    x_data = npu.pad(x.data, padding)
    x_data = npu.unfold(x_data, w.shape[-2:], stride, dilation)  # type: ignore

    out_data = np.einsum("nixyhw,oihw->noxy", x_data, w.data, optimize=_EINSUM_OPTIM)

    prevs = tuple(p for p in (x, w) if p.requires_grad) if is_grad_enabled() else None

    if prevs:
        backward = lambda out: _conv2d_backward(out, x, w, stride, padding, dilation)
    else:
        backward = None

    out = out_array(out_data, prevs, backward)

    return out if b is None else out + np.expand_dims(b, (0, -2, -1))


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
