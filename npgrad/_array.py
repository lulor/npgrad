from __future__ import annotations

from copy import copy
from typing import Callable

import numpy as np
from numpy.typing import ArrayLike, DTypeLike, NDArray

from npgrad._grad import is_grad_enabled
from npgrad.typing import ShapeLike

_handled_functions = {}


def implements(np_function):
    """
    Register an __array_ufunc__/__array_function__ implementation for Array objects.
    """

    def decorator(func):
        _handled_functions[np_function] = func
        return func

    return decorator


def _get_item_backward(out: Array, x: Array, key) -> None:
    assert out.grad is not None
    if x.requires_grad:
        assert x.grad is not None
        x.grad[key] += out.grad


class Array:
    def __init__(
        self,
        data: ArrayLike,
        dtype: DTypeLike = None,
        requires_grad: bool = False,
        _prevs: tuple[Array, ...] | None = None,
        _backward: Callable[[Array], None] | None = None,
    ) -> None:
        assert bool(_prevs) == bool(_backward), "_prevs and _backward mismatch"
        assert not _prevs or requires_grad, "non-leaf arrays must require grad"
        self._data = np.asarray(data, dtype=dtype)
        self._grad = None
        grad_enabled = is_grad_enabled()
        self._requires_grad = requires_grad and grad_enabled
        self._prevs = frozenset(_prevs) if _prevs and grad_enabled else None
        self._backward = _backward if grad_enabled else None
        self._retains_grad = False

    def __array__(self, dtype=None, copy=None) -> NDArray:
        if dtype is not None and dtype != self.data.dtype:
            if copy is False:
                raise ValueError(
                    f"a copy is required to cast from {self.data.dtype} to {dtype}"
                )
            return self.data.astype(dtype)
        elif copy:
            return self.data.copy()
        return self.data

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if ufunc not in _handled_functions or method != "__call__":
            return NotImplemented
        return _handled_functions[ufunc](*inputs, **kwargs)

    def __array_function__(self, func, types, args, kwargs):
        if func not in _handled_functions:
            return NotImplemented
        # Note: this allows subclasses that don't override
        # __array_function__ to handle Array objects.
        if not all(issubclass(t, self.__class__) for t in types):
            return NotImplemented
        return _handled_functions[func](*args, **kwargs)

    def _check_array_assignment(self, value: NDArray, property: str) -> None:
        if value.shape != self.shape:
            raise ValueError(
                f"cannot assign {property} with shape {value.shape} to array with shape {self.shape}"
            )
        if value.dtype != self.dtype:
            raise ValueError(
                f"cannot assign {property} with dtype {value.dtype} to array with dtype {self.dtype}"
            )

    @property
    def data(self) -> NDArray:
        return self._data

    @data.setter
    def data(self, value: NDArray) -> None:
        self._check_array_assignment(value, "data")
        self._data = value

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def size(self) -> int:
        return self.data.size

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def dtype(self) -> np.dtype:
        return self.data.dtype

    @property
    def grad(self) -> NDArray | None:
        return self._grad

    @grad.setter
    def grad(self, value: NDArray | None) -> None:
        if value is not None:
            self._check_array_assignment(value, "grad")
        self._grad = value

    @property
    def is_leaf(self) -> bool:
        return not self._prevs

    @property
    def requires_grad(self) -> bool:
        return self._requires_grad

    @requires_grad.setter
    def requires_grad(self, value: bool) -> None:
        if not self.is_leaf:
            raise RuntimeError("can only change requires_grad for leaf arrays")
        self._requires_grad = value

    def requires_grad_(self, requires_grad: bool = True) -> Array:
        self.requires_grad = requires_grad
        return self

    @property
    def retains_grad(self) -> bool:
        return self._retains_grad

    def retain_grad(self) -> None:
        self._retains_grad = True

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, key) -> Array:
        if self.requires_grad:
            prevs = (self,)
            key = copy(key)
            backward = lambda out: _get_item_backward(out, self, key)
        else:
            prevs = backward = None
        return Array(
            self.data[key],
            requires_grad=self.requires_grad,
            _prevs=prevs,
            _backward=backward,
        )

    def __repr__(self) -> str:
        return repr(self.data).replace("array", type(self).__name__)

    @property
    def T(self) -> Array:
        return self.transpose()

    def item(self) -> float:
        return self.data.item()

    def detach(self) -> Array:
        return Array(self.data)

    def backward(self) -> None:
        if not self.requires_grad:
            raise RuntimeError("cannot call backward() on array not requiring grad")

        def visit(n: Array) -> None:
            if n not in visited:
                visited.add(n)
                if n._prevs:  # == if not n.is_leaf
                    for prev in n._prevs:
                        visit(prev)
                    topo.append(n)  # only append non-leaf nodes
                if n.requires_grad and n.grad is None:
                    n.grad = np.zeros_like(n.data)

        topo: list[Array] = []
        visited = set()
        self.grad = np.ones_like(self.data)
        visit(self)

        for n in reversed(topo):
            assert n._backward is not None
            n._backward(n)
            if not n.retains_grad:
                n.grad = None

    def __add__(self, other: ArrayLike) -> Array:
        return np.add(self, other)  # type: ignore

    __radd__ = __add__

    def __mul__(self, other: ArrayLike) -> Array:
        return np.multiply(self, other)  # type: ignore

    __rmul__ = __mul__

    def __matmul__(self, other: ArrayLike) -> Array:
        return np.matmul(self, other)

    def __rmatmul__(self, other: ArrayLike) -> Array:
        return np.matmul(other, self)

    def __neg__(self) -> Array:
        return np.negative(self)  # type: ignore

    def __pow__(self, other: ArrayLike) -> Array:
        return np.power(self, other)  # type: ignore

    def __sub__(self, other: ArrayLike) -> Array:
        return np.subtract(self, other)  # type: ignore

    def __rsub__(self, other: ArrayLike) -> Array:
        return np.subtract(other, self)  # type: ignore

    def __truediv__(self, other: ArrayLike) -> Array:
        return np.divide(self, other)  # type: ignore

    def __rtruediv__(self, other: ArrayLike) -> Array:
        return np.divide(other, self)  # type: ignore

    def reshape(self, newshape: ShapeLike) -> Array:
        return np.reshape(self, newshape)  # type: ignore

    def squeeze(self, axis: ShapeLike | None = None) -> Array:
        return np.squeeze(self, axis)  # type: ignore

    def transpose(self, axes: ShapeLike | None = None) -> Array:
        return np.transpose(self, axes)  # type: ignore

    def sum(self, axis: ShapeLike | None = None, keepdims: bool = False) -> Array:
        return np.sum(self, axis, keepdims=keepdims)

    def mean(self, axis: ShapeLike | None = None, keepdims: bool = False) -> Array:
        return np.mean(self, axis, keepdims=keepdims)

    def max(self, axis: ShapeLike | None = None, keepdims: bool = False) -> Array:
        return np.amax(self, axis, keepdims=keepdims)

    def min(self, axis: ShapeLike | None = None, keepdims: bool = False) -> Array:
        return np.amin(self, axis, keepdims=keepdims)


def asarray_(data: ArrayLike) -> Array:
    """
    Wrap the input object in an array if necessary.

    Parameters
    ----------
    data : array_like
        The object to convert into an array.
        If data is already an array with requires_grad=True, then data itself is returned.
        Otherwise, a new wrapper array will be created.
        In any case, if data is already an array/ndarray, the returned object will share
        the same underlying ndarray with it (i.e., no copy operation is performed).

    Returns
    -------
    out : array
        The input object in case it is already an array with requires_grad=True,
        otherwise a new wrapper array containing the input data.
    """
    return data if isinstance(data, Array) and data.requires_grad else Array(data)


def array(
    data: ArrayLike, dtype: DTypeLike = None, requires_grad: bool = False
) -> Array:
    """
    Build a new array.

    Parameters
    ----------
    data : array_like
        The elements of the new array.
    dtype : dtype, optional
        The dtype if the new array.
    require_grad : bool, optional
        Whether the new array requires grad.

    Returns
    -------
    out : array
        The new array.
    """
    return Array(np.array(data, dtype=dtype), requires_grad=requires_grad)


def asarray(data: ArrayLike, dtype: DTypeLike = None) -> Array:
    """
    Convert the input data to an array.

    Parameters
    ----------
    data : array_like
        The elements to convert.
    dtype : dtype, optional
        If None (default) or matching the input's dtype, the input itself is returned.

    Returns
    -------
    out : array
        The input data if it is already an Array with matching dtype, otherwise a new
        array containing the input data. No copy operation is performed unless necessary
        for a dtype conversion.
    """
    return (
        data
        if isinstance(data, Array) and (dtype is None or dtype == data.dtype)
        else Array(data, dtype=dtype)
    )
