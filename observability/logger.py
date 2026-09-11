from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class JsonlObservabilityLogger:
    def __init__(self, base_dir: str = "observability/logs", environment: str = "dev") -> None:
        self.log_dir = Path(base_dir) / environment
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def _get_file_path(self) -> Path:
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return self.log_dir / f"agent_events_{date_str}.jsonl"

    def log_event(self, event: dict[str, Any]) -> None:
        if "timestamp" not in event:
            event["timestamp"] = datetime.now(timezone.utc).isoformat()

        file_path = self._get_file_path()

        with file_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")