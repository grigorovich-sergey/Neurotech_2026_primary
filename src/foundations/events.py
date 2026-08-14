"""Minimal structured JSON Lines event logging."""

from dataclasses import asdict, dataclass, field
import json
import math
from numbers import Real
from pathlib import Path
import threading
from typing import Any


@dataclass(frozen=True)
class Event:
    timestamp: float
    name: str
    payload: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.timestamp, bool) or not isinstance(self.timestamp, Real):
            raise TypeError("event timestamp must be a real number")
        if not math.isfinite(float(self.timestamp)) or self.timestamp < 0:
            raise ValueError("event timestamp must be finite and non-negative")
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("event name must be a non-empty string")
        if not isinstance(self.payload, dict):
            raise TypeError("event payload must be a dict")
        try:
            json.dumps(self.payload, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("event payload must be JSON serializable") from exc


class JsonlEventLogger:
    """Append events to one JSON object per line."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")
        self._lock = threading.Lock()

    def log(self, event: Event) -> None:
        line = json.dumps(
            asdict(event),
            allow_nan=False,
            separators=(",", ":"),
        ) + "\n"
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line)


if __name__ == "__main__":
    print(Event(0.0, "smoke_check", {"ok": True}))
