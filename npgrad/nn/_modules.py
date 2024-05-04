from __future__ import annotations

from typing import Any, Callable, Iterator

import numpy as np
from numpy.typing import ArrayLike

from .._array import Array
from . import functional as F
from ._parameter import Parameter
from ._utils import pair

_DEFAULT_DTYPE = np.float32

__all__ = [
    "Module",
    "ReLU",
    "Linear",
    "Conv2d",
    "MaxPool2d",
    "AvgPool2d",
    "Softmax",
    "CrossEntropyLoss",
]


def _forward_unimplemented(self, *_, **__) -> None:
    raise NotImplementedError(
        f"Module [{type(self).__name__}] is missing the required 'forward' function"
    )


class Module:

    forward: Callable[..., Any] = _forward_unimplemented

    _parameters: dict[str, Parameter]
    _modules: dict[str, Module]

    def __init__(self) -> None:
        super().__setattr__("_parameters", {})
        super().__setattr__("_modules", {})

    def __getattr__(self, name: str) -> Any:
        if name in self._parameters:
            return self._parameters[name]
        if name in self._modules:
            return self._modules[name]
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )

    def __delattr__(self, name: str) -> None:
        if name in self._parameters:
            del self._parameters[name]
        elif name in self._modules:
            del self._modules[name]
        else:
            super().__delattr__(name)

    def __setattr__(self, name: str, value: Any) -> None:
        # Try to delete the object's attr before setting the new value.
        # Note that this raises 'AttributeError' even in cases in which
        # 'getattr(self, name)' would succeed (e.g. when accessing class attributes),
        # so it's important not to use 'getattr'/'hasattr' in the 'try' block.
        try:
            delattr(self, name)
        except AttributeError:
            pass
        if isinstance(value, Parameter):
            self._parameters[name] = value
        elif isinstance(value, Module):
            self._modules[name] = value
        else:
            super().__setattr__(name, value)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)

    def modules(self) -> Iterator[Module]:
        yield self
        for m in self._modules.values():
            yield from m.modules()

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        modules = self.modules() if recurse else (self,)
        for m in modules:
            yield from m._parameters.values()

    def requires_grad(self, requires_grad: bool = True) -> None:
        for p in self.parameters():
            p.requires_grad = requires_grad


class ReLU(Module):
    def forward(self, x: ArrayLike) -> Array:
        return F.relu(x)


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(
            np.zeros((in_features, out_features), dtype=_DEFAULT_DTYPE)
        )
        self.bias = (
            Parameter(np.zeros(out_features, dtype=_DEFAULT_DTYPE)) if bias else None
        )
        self.requires_grad()

    def forward(self, x: ArrayLike) -> Array:
        x = x @ self.weight
        return x if self.bias is None else x + self.bias


class Conv2d(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = pair(kernel_size)
        self.stride = pair(stride)
        self.padding = pair(padding)
        self.dilation = pair(dilation)
        self.weight = Parameter(
            np.zeros(
                (out_channels, in_channels, *self.kernel_size), dtype=_DEFAULT_DTYPE
            )
        )
        self.bias = (
            Parameter(np.zeros(out_channels, dtype=_DEFAULT_DTYPE)) if bias else None
        )
        self.requires_grad()

    def forward(self, x: ArrayLike) -> Array:
        return F.conv2d(
            x, self.weight, self.bias, self.stride, self.padding, self.dilation
        )


class MaxPool2d(Module):
    def __init__(
        self,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] | None = None,
        padding: int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
    ) -> None:
        super().__init__()
        self.kernel_size = pair(kernel_size)
        self.stride = pair(kernel_size if stride is None else stride)
        self.padding = pair(padding)
        self.dilation = pair(dilation)

    def forward(self, x: ArrayLike) -> Array:
        return F.max_pool2d(
            x, self.kernel_size, self.stride, self.padding, self.dilation
        )


class AvgPool2d(Module):
    def __init__(
        self,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] | None = None,
        padding: int | tuple[int, int] = 0,
    ) -> None:
        super().__init__()
        self.kernel_size = pair(kernel_size)
        self.stride = pair(kernel_size if stride is None else stride)
        self.padding = pair(padding)

    def forward(self, x: ArrayLike) -> Array:
        return F.avg_pool2d(x, self.kernel_size, self.stride, self.padding)


class Softmax(Module):
    def __init__(self, axis: int) -> None:
        super().__init__()
        self.axis = axis

    def forward(self, x: ArrayLike) -> Array:
        return F.softmax(x, self.axis)


class CrossEntropyLoss(Module):
    def forward(self, x: ArrayLike, target: ArrayLike) -> Array:
        return F.cross_entropy(x, target)
