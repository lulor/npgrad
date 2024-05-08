from contextlib import contextmanager

_grad_enabled = True


def is_grad_enabled() -> bool:
    return _grad_enabled


def set_grad_enabled(enabled: bool) -> None:
    global _grad_enabled
    _grad_enabled = enabled


@contextmanager
def no_grad():
    prev = is_grad_enabled()
    set_grad_enabled(False)
    yield
    set_grad_enabled(prev)
