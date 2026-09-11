from typing import Dict, Any, List

from core.types import TraceEvent


class Tracer:
    def __init__(self):
        self.events: List[TraceEvent] = []

    def add(self, event: str, **data: Any) -> None:
        self.events.append(TraceEvent(event=event, data=dict(data)))

    def as_dict_list(self) -> List[Dict[str, Any]]:
        return [{"event": e.event, **e.data} for e in self.events]
