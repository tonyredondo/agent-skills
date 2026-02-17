#!/usr/bin/env python3
from __future__ import annotations

"""Structured logger with heartbeat and timing helpers."""

import json
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Dict, Iterator, Optional

from .config import LoggingConfig


LEVEL_TO_INT = {
    "DEBUG": 10,
    "INFO": 20,
    "WARN": 30,
    "ERROR": 40,
}


def _safe_json(value: object) -> str:
    """Serialize log fields deterministically."""
    return json.dumps(value, ensure_ascii=True, sort_keys=True)


@dataclass
class Logger:
    """Minimal structured logger used across pipeline modules."""

    config: LoggingConfig
    run_id: str

    @staticmethod
    def create(config: LoggingConfig) -> "Logger":
        """Create logger with short random run id."""
        return Logger(config=config, run_id=uuid.uuid4().hex[:10])

    def _enabled(self, level: str) -> bool:
        """Check whether target level should be emitted."""
        current = LEVEL_TO_INT.get(self.config.level, 20)
        wanted = LEVEL_TO_INT.get(level, 20)
        return wanted >= current

    def _emit(self, level: str, message: str, fields: Optional[Dict[str, object]] = None) -> None:
        """Emit one structured log line to stderr."""
        if not self._enabled(level):
            return
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        event_id = uuid.uuid4().hex[:8] if self.config.include_event_ids else ""
        suffix = ""
        if fields:
            suffix = " " + _safe_json(fields)
        if event_id:
            line = f"[{timestamp}] [{level}] [run:{self.run_id}] [event:{event_id}] {message}{suffix}"
        else:
            line = f"[{timestamp}] [{level}] [run:{self.run_id}] {message}{suffix}"
        print(line, file=sys.stderr, flush=True)

    def debug(self, message: str, **fields: object) -> None:
        """Emit DEBUG logs only when debug events are enabled."""
        if self.config.debug_events:
            self._emit("DEBUG", message, fields or None)

    def info(self, message: str, **fields: object) -> None:
        """Emit INFO log line."""
        self._emit("INFO", message, fields or None)

    def warn(self, message: str, **fields: object) -> None:
        """Emit WARN log line."""
        self._emit("WARN", message, fields or None)

    def error(self, message: str, **fields: object) -> None:
        """Emit ERROR log line."""
        self._emit("ERROR", message, fields or None)

    @contextmanager
    def timed(self, name: str, **fields: object) -> Iterator[None]:
        """Context manager that logs start/end with elapsed milliseconds."""
        started = time.time()
        self.info(f"{name} started", **fields)
        try:
            yield
        finally:
            elapsed_ms = int((time.time() - started) * 1000)
            end_fields = dict(fields)
            end_fields["elapsed_ms"] = elapsed_ms
            self.info(f"{name} completed", **end_fields)

    @contextmanager
    def heartbeat(
        self,
        label: str,
        status_fn: Optional[Callable[[], Dict[str, object]]] = None,
    ) -> Iterator[None]:
        """Background heartbeat context manager for long-running stages."""
        stop = threading.Event()
        interval = max(1, int(self.config.heartbeat_seconds))

        def loop() -> None:
            while not stop.wait(interval):
                payload: Dict[str, object] = {"label": label}
                if status_fn is not None:
                    try:
                        payload.update(status_fn())
                    except Exception as exc:  # pragma: no cover - debug path
                        payload["status_error"] = str(exc)
                self.info("heartbeat", **payload)

        thread = threading.Thread(target=loop, name=f"hb-{label}", daemon=True)
        thread.start()
        try:
            yield
        finally:
            stop.set()
            thread.join(timeout=interval)

