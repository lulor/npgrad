__all__ = ["Optimizer", "SGD"]

from typing import Iterable

from numpy.typing import NDArray

from npgrad._array import Array


class Optimizer:
    def __init__(self, params: Iterable[Array], lr: float) -> None:
        self._params = tuple(params)
        self._lr = lr

    def zero_grad(self) -> None:
        for p in self._params:
            p.grad = None

    def step(self) -> None:
        raise NotImplementedError


class SGD(Optimizer):
    _momentum_buf: list[NDArray | None]

    def __init__(
        self,
        params: Iterable[Array],
        lr: float = 1e-3,
        momentum: float = 0,
        dampening: float = 0,
        weight_decay: float = 0,
        nesterov: bool = False,
    ) -> None:
        super().__init__(params, lr)
        self._momentum = momentum
        self._dampening = dampening
        self._weight_decay = weight_decay
        self._nesterov = nesterov
        self._momentum_buf = [None] * len(self._params) if momentum else []

    def step(self) -> None:
        _sgd(
            self._params,
            self._momentum_buf,
            self._lr,
            self._momentum,
            self._dampening,
            self._weight_decay,
            self._nesterov,
        )


def _sgd(
    params: Iterable[Array],
    momentum_buf: list[NDArray | None],
    lr: float,
    momentum: float,
    dampening: float,
    weight_decay: float,
    nesterov: bool,
) -> None:
    for i, p in enumerate(params):
        if p.grad is not None:
            d_p = p.grad

            if weight_decay:
                # do not modify d_p in-place
                d_p = d_p + weight_decay * p.data

            if momentum:
                buf = momentum_buf[i]

                if buf is None:
                    buf = momentum_buf[i] = d_p.copy() if d_p is p.grad else d_p
                else:
                    buf *= momentum
                    buf += (1 - dampening) * d_p

                if nesterov:
                    # do not modify d_p in-place
                    d_p = d_p + momentum * buf
                else:
                    d_p = buf

            p.data -= lr * d_p
