__all__ = ["Parameter"]

from npgrad._array import Array


class Parameter(Array):
    def __repr__(self) -> str:
        return repr(self.data).replace("array", "Param")
