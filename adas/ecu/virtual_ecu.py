from __future__ import annotations

from adas.ecu.can_messages import (
    FCW_ID, LDW_ID, CanFrame, decode_fcw, decode_ldw,
)


class DashboardEcu:
    """Simulated dashboard ECU: decodes alert frames and checks their integrity."""

    def __init__(self) -> None:
        self._last_counter: dict[int, int] = {}
        self.rejected = 0

    def receive(self, frame: CanFrame) -> str:
        """Process one frame and return the text the dashboard would show."""
        try:
            if frame.arbitration_id == FCW_ID:
                m = decode_fcw(frame)
                ttc = f"{m.ttc_s:.1f} s" if m.ttc_s is not None else "n/a"
                text = f"COLLISION WARNING {m.level.upper()}  object #{m.track_id}  TTC {ttc}"
            elif frame.arbitration_id == LDW_ID:
                m = decode_ldw(frame)
                text = f"LANE DEPARTURE {m.side.upper()}  offset {m.offset:+.2f}"
            else:
                raise ValueError(f"unknown id 0x{frame.arbitration_id:X}")
        except ValueError as exc:
            self.rejected += 1
            return f"REJECTED ({exc})"
        return text + self._check_counter(frame.arbitration_id, m.counter)

    def _check_counter(self, arbitration_id: int, counter: int) -> str:
        last = self._last_counter.get(arbitration_id)
        self._last_counter[arbitration_id] = counter
        if last is None:
            return ""
        if counter == last:
            return "  [warning: repeated counter, sender may be frozen]"
        lost = (counter - last - 1) % 256
        return f"  [warning: {lost} message(s) lost]" if lost else ""