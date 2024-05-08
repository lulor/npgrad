__all__ = ["Parameter"]

from numpy.typing import ArrayLike, DTypeLike

from npgrad._array import Array


class Parameter(Array):
    def __init__(
        self,
        data: ArrayLike,
        dtype: DTypeLike = None,
        requires_grad: bool = True,
    ) -> None:
        super().__init__(data, dtype)
        # set requires_grad even if grad is globally disabled
        self.requires_grad = requires_grad

    def __repr__(self) -> str:
        return repr(self.data).replace("array", "Param")
