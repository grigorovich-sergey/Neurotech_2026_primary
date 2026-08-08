"""Deterministic participant-level G/E session counterbalancing."""

from dataclasses import dataclass

from experiment_learning.contracts import Condition


@dataclass
class SessionSchedule:
    """ABAB/BABA schedule selected by pseudonymous participant sequence parity."""

    participant_sequence_index: int
    next_session_index: int = 0

    def __post_init__(self) -> None:
        for name in ("participant_sequence_index", "next_session_index"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")

    def condition_for(self, session_index: int) -> Condition:
        if isinstance(session_index, bool) or not isinstance(session_index, int) or session_index < 0:
            raise ValueError("session_index must be a non-negative integer")
        g_first = self.participant_sequence_index % 2 == 0
        use_g = (session_index % 2 == 0) == g_first
        return Condition.G if use_g else Condition.E

    def allocate_next(self) -> tuple[int, Condition]:
        index = self.next_session_index
        condition = self.condition_for(index)
        self.next_session_index += 1
        return index, condition


if __name__ == "__main__":
    for participant_index in (0, 1):
        schedule = SessionSchedule(participant_index)
        print(participant_index, [schedule.allocate_next()[1].value for _ in range(4)])
