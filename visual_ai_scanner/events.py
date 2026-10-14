"""Track lifecycle events without duplicating one-per-frame."""
from __future__ import annotations

import time
from collections import OrderedDict, deque
from dataclasses import dataclass


@dataclass
class Event:
    kind: str          # appear | disappear | first_seen
    track_id: int | None
    label: str
    timestamp: float
    score: float | None = None


class EventHistory:
    def __init__(self, max_events: int = 500, disappear_grace_s: float = 1.5):
        self.max_events = max_events
        self.grace = disappear_grace_s
        self.events: deque[Event] = deque(maxlen=max_events)
        self._last_seen: "OrderedDict[int, tuple[float, str]]" = OrderedDict()

    def update(self, detections, now: float | None = None) -> list[Event]:
        now = now if now is not None else time.time()
        new_events: list[Event] = []
        seen_ids: set[int] = set()

        for d in detections:
            if d.track_id is None:
                continue
            tid = d.track_id
            seen_ids.add(tid)
            if tid not in self._last_seen:
                ev = Event("appear", tid, d.label, now, d.score)
                self.events.append(ev)
                new_events.append(ev)
            self._last_seen[tid] = (now, d.label)
            self._last_seen.move_to_end(tid)

        # Detect disappearances
        for tid in list(self._last_seen.keys()):
            if tid in seen_ids:
                continue
            last_t, label = self._last_seen[tid]
            if now - last_t > self.grace:
                ev = Event("disappear", tid, label, now)
                self.events.append(ev)
                new_events.append(ev)
                del self._last_seen[tid]

        return new_events