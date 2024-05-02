from typing import Iterable

from ._array import Array

__all__ = ["Optimizer", "SGD"]


class Optimizer:
    def __init__(self, params: Iterable[Array], lr: float) -> None:
        self._params = tuple(params)
        self.lr = lr

    @property
    def params(self) -> tuple[Array, ...]:
        return self._params

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None

    def step(self) -> None:
        raise NotImplementedError


class SGD(Optimizer):
    def step(self) -> None:
        for p in self.params:
            if p.requires_grad:
                if p.grad is None:
                    raise RuntimeError("parameter gradient is None")
                p.data -= self.lr * p.grad
