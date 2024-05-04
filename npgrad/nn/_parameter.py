from .._array import Array

__all__ = ["Parameter"]


class Parameter(Array):
    def __repr__(self) -> str:
        return repr(self.data).replace("array", "Param")
