from typing import Any, Iterator

import numpy as np
import numpy.typing as npt

from ..array import Array
from . import functional as F
from .utils import as_tuples

_DEFAULT_DTYPE = np.float32


class Module:
    def parameters(self) -> Iterator[Array]:
        params = []
        for var in vars(self).values():
            if isinstance(var, Module):
                params.extend(var.parameters())
            elif isinstance(var, Array):
                params.append(var)
        return iter(params)

    def requires_grad(self, requires_grad: bool = True) -> None:
        for p in self.parameters():
            p.requires_grad = requires_grad

    def __call__(self, *args, **kwargs) -> Any:
        return self.forward(*args, **kwargs)

    def forward(self, *_, **__) -> None:
        raise NotImplementedError(
            f'Module [{type(self).__name__}] is missing the required "forward" function'
        )


class ReLU(Module):
    def forward(self, x: npt.ArrayLike) -> Array:
        return F.relu(x)


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Array(np.zeros((in_features, out_features), dtype=_DEFAULT_DTYPE))
        self.bias = (
            Array(np.zeros(out_features, dtype=_DEFAULT_DTYPE)) if bias else None
        )
        self.requires_grad()

    def forward(self, x: npt.ArrayLike) -> Array:
        x = x @ self.weight
        return x + self.bias if self.bias is not None else x


class Conv2d(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        bias: bool = False,
    ) -> None:
        super().__init__()
        kernel_size, stride, padding, dilation = as_tuples(
            kernel_size, stride, padding, dilation
        )
        if bias:
            raise NotImplementedError
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.weight = Array(
            np.zeros((out_channels, in_channels, *kernel_size), dtype=_DEFAULT_DTYPE)
        )
        self.requires_grad()

    def forward(self, x: npt.ArrayLike) -> Array:
        return F.conv2d(x, self.weight, self.stride, self.padding, self.dilation)


class MaxPool2d(Module):
    def __init__(
        self,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] | None = None,
        padding: int | tuple[int, int] = 0,
    ) -> None:
        super().__init__()
        stride = kernel_size if stride is None else stride
        kernel_size, stride, padding = as_tuples(kernel_size, stride, padding)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x: npt.ArrayLike) -> Array:
        return F.max_pool2d(x, self.kernel_size, self.stride, self.padding)


class AvgPool2d(Module):
    def __init__(
        self,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] | None = None,
        padding: int | tuple[int, int] = 0,
    ) -> None:
        super().__init__()
        stride = kernel_size if stride is None else stride
        kernel_size, stride, padding = as_tuples(kernel_size, stride, padding)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x: npt.ArrayLike) -> Array:
        return F.avg_pool2d(x, self.kernel_size, self.stride, self.padding)


class Softmax(Module):
    def __init__(self, axis: int) -> None:
        super().__init__()
        self.axis = axis

    def forward(self, x: npt.ArrayLike) -> Array:
        return F.softmax(x, self.axis)


class CrossEntropyLoss(Module):
    def forward(self, x: npt.ArrayLike, target: npt.ArrayLike) -> Array:
        return F.cross_entropy(x, target)
