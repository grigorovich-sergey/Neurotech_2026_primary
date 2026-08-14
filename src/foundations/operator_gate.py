"""Interactive operator gates with refreshable setup status."""

from collections.abc import Callable
import math
from numbers import Real
import os
import sys
import time


_START_KEY = " "
_ABORT_KEYS = {"q", "Q", "\x1b", "\x03"}


def format_impedance(value: float | None) -> str:
    """Format one Guardian impedance reading for a fitting display."""

    if value is None:
        return "waiting for first reading"
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise ValueError("impedance must be a finite non-negative number or None")
    ohms = float(value)
    if ohms < 1_000.0:
        return f"{ohms:.0f} ohm"
    return f"{ohms / 1_000.0:.1f} kOhm ({ohms:.0f} ohm)"


def _decision(key: str) -> bool | None:
    if key == _START_KEY:
        return True
    if key in _ABORT_KEYS:
        return False
    return None


def _wait_with_reader(
    read_key: Callable[[], str],
    *,
    status: Callable[[], str] | None,
    emit: Callable[[str], None] | None,
) -> bool:
    previous: str | None = None
    while True:
        if status is not None:
            current = status()
            if not isinstance(current, str) or not current:
                raise ValueError("operator-gate status must be a non-empty string")
            if emit is not None and current != previous:
                emit(current)
            previous = current
        result = _decision(read_key())
        if result is not None:
            return result


def _wait_windows(
    *,
    status: Callable[[], str] | None,
    emit: Callable[[str], None] | None,
    refresh_seconds: float,
) -> bool:
    import msvcrt

    previous: str | None = None
    while True:
        if status is not None:
            current = status()
            if not isinstance(current, str) or not current:
                raise ValueError("operator-gate status must be a non-empty string")
            if emit is not None and current != previous:
                emit(current)
            previous = current
        deadline = time.monotonic() + refresh_seconds
        while time.monotonic() < deadline:
            if msvcrt.kbhit():
                result = _decision(msvcrt.getwch())
                if result is not None:
                    return result
            time.sleep(min(0.02, refresh_seconds))


def _wait_posix(
    *,
    status: Callable[[], str] | None,
    emit: Callable[[str], None] | None,
    refresh_seconds: float,
) -> bool:
    if not sys.stdin.isatty():
        raise RuntimeError("the operator SPACE gate requires an interactive terminal")

    import select
    import termios
    import tty

    descriptor = sys.stdin.fileno()
    previous_terminal = termios.tcgetattr(descriptor)
    previous_status: str | None = None
    try:
        tty.setcbreak(descriptor)
        while True:
            if status is not None:
                current = status()
                if not isinstance(current, str) or not current:
                    raise ValueError("operator-gate status must be a non-empty string")
                if emit is not None and current != previous_status:
                    emit(current)
                previous_status = current
            readable, _, _ = select.select([descriptor], [], [], refresh_seconds)
            if not readable:
                continue
            result = _decision(sys.stdin.read(1))
            if result is not None:
                return result
    finally:
        termios.tcsetattr(descriptor, termios.TCSADRAIN, previous_terminal)


def wait_for_space_or_abort(
    *,
    status: Callable[[], str] | None = None,
    emit: Callable[[str], None] | None = None,
    refresh_seconds: float = 0.25,
    read_key: Callable[[], str] | None = None,
) -> bool:
    """Refresh setup status until SPACE starts or Q/Esc/Ctrl-C aborts."""

    if status is not None and not callable(status):
        raise TypeError("status must be callable or None")
    if emit is not None and not callable(emit):
        raise TypeError("emit must be callable or None")
    if (
        isinstance(refresh_seconds, bool)
        or not isinstance(refresh_seconds, Real)
        or not math.isfinite(float(refresh_seconds))
        or float(refresh_seconds) <= 0.0
    ):
        raise ValueError("refresh_seconds must be a positive finite number")
    if read_key is not None:
        if not callable(read_key):
            raise TypeError("read_key must be callable or None")
        return _wait_with_reader(read_key, status=status, emit=emit)
    if os.name == "nt":
        return _wait_windows(
            status=status,
            emit=emit,
            refresh_seconds=float(refresh_seconds),
        )
    return _wait_posix(
        status=status,
        emit=emit,
        refresh_seconds=float(refresh_seconds),
    )


if __name__ == "__main__":
    print(format_impedance(12_000.0))
