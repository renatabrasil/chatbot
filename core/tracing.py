from typing import Dict, Any, List


class Tracer:
    def __init__(self):
        self.events: List[Dict[str, Any]] = []

    def add(self, event_type: str, **kwargs):
        event = {"event": event_type}
        event.update(kwargs)
        self.events.append(event)
