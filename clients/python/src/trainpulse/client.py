"""Zero-dependency client for the TrainPulse diagnostics daemon.

Minimal integration is two lines around the training step::

    tp = TrainPulseClient()
    ...
    with tp.step(global_step=step, tokens=tokens_in_batch):
        loss = train_step(batch)

Reporting never raises by default (``fail_silently=True``): a monitoring
sidecar must not be able to crash a training run.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional

__all__ = ["TrainPulseClient", "Tuner", "TrainPulseConnectionError"]

_DEFAULT_URL = "http://127.0.0.1:9876"


class TrainPulseConnectionError(Exception):
    """The daemon could not be reached or answered with an error."""


class TrainPulseClient:
    def __init__(
        self,
        base_url: str = _DEFAULT_URL,
        token: Optional[str] = None,
        timeout: float = 2.0,
        fail_silently: bool = True,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout
        self.fail_silently = fail_silently

    # -- write path ---------------------------------------------------------

    def report(self, **metrics: Any) -> bool:
        """POST training-loop metrics to ``/v1/training``.

        Keyword names mirror the daemon's JSON fields: ``step_time_ms``,
        ``tokens_per_sec``, ``mfu``, ``global_step``, ``data_wait_ms`` ...
        Returns True when the daemon accepted the sample.
        """
        return self._post("/v1/training", metrics) is not None

    def report_framework(self, name: str, payload: Dict[str, Any]) -> bool:
        """POST framework-native metrics for server-side normalization."""
        return self._post(f"/v1/framework?name={name}", payload) is not None

    @contextmanager
    def step(
        self,
        global_step: Optional[int] = None,
        tokens: Optional[int] = None,
        **extra: Any,
    ) -> Iterator[None]:
        """Time one training step and report it on exit."""
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            metrics: Dict[str, Any] = dict(extra)
            metrics["step_time_ms"] = elapsed_ms
            if global_step is not None:
                metrics["global_step"] = global_step
            if tokens is not None and elapsed_ms > 0:
                metrics["tokens_per_sec"] = tokens / (elapsed_ms / 1000.0)
            self.report(**metrics)

    # -- read path ----------------------------------------------------------

    def snapshot(self) -> Dict[str, Any]:
        """Full daemon snapshot: telemetry, signals, diagnoses, utilization."""
        return self._get("/v1/snapshot", fail_silently=False)

    def health(self) -> float:
        """Current health score, 0-100."""
        return float(self.snapshot().get("health", 0.0))

    def recommendations(self) -> Dict[str, Any]:
        """Cluster utilization plus active optimization recommendations."""
        return self._get("/v1/recommendations", fail_silently=False)

    def version(self) -> Dict[str, Any]:
        return self._get("/v1/version", fail_silently=False)

    # -- plumbing -----------------------------------------------------------

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def _get(self, path: str, fail_silently: Optional[bool] = None) -> Any:
        req = urllib.request.Request(self.base_url + path, headers=self._headers())
        return self._send(req, fail_silently)

    def _post(self, path: str, payload: Dict[str, Any]) -> Any:
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.base_url + path, data=body, headers=self._headers(), method="POST"
        )
        return self._send(req, None)

    def _send(self, req: urllib.request.Request, fail_silently: Optional[bool]) -> Any:
        silent = self.fail_silently if fail_silently is None else fail_silently
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read()
                if not raw:
                    return {}
                try:
                    return json.loads(raw)
                except json.JSONDecodeError:
                    return {"raw": raw.decode("utf-8", "replace")}
        except (urllib.error.URLError, urllib.error.HTTPError, OSError) as exc:
            if silent:
                return None
            raise TrainPulseConnectionError(str(exc)) from exc


class Tuner:
    """Consult the daemon's optimizer from inside the training loop.

    Only recommendations the daemon marks ``auto_applicable`` (convergence-
    neutral knobs such as dataloader worker counts) are surfaced through
    :meth:`suggested`; everything else stays advisory in :meth:`advisories`.

        tuner = Tuner(client)
        workers = tuner.suggested("dataloader_workers", current=workers)
    """

    def __init__(self, client: TrainPulseClient, refresh_seconds: float = 10.0) -> None:
        self.client = client
        self.refresh_seconds = refresh_seconds
        self._cache: List[Dict[str, Any]] = []
        self._fetched_at = 0.0

    def _refresh(self) -> None:
        if time.monotonic() - self._fetched_at < self.refresh_seconds:
            return
        self._fetched_at = time.monotonic()
        try:
            data = self.client.recommendations()
        except TrainPulseConnectionError:
            return  # keep the previous cache; monitoring must not break training
        self._cache = data.get("recommendations") or []

    def recommendations(self) -> List[Dict[str, Any]]:
        self._refresh()
        return list(self._cache)

    def advisories(self) -> List[Dict[str, Any]]:
        """Recommendations that need a human decision (not auto-applicable)."""
        return [r for r in self.recommendations() if not r.get("auto_applicable")]

    def recommendation_for(self, parameter: str, auto_only: bool = True) -> Optional[Dict[str, Any]]:
        """Return the active recommendation targeting ``parameter``, if any.

        Useful when the suggestion is directional rather than numeric::

            if tuner.recommendation_for("dataloader_workers"):
                workers = min(workers + 2, max_workers)
        """
        for rec in self.recommendations():
            if rec.get("parameter") != parameter:
                continue
            if auto_only and not rec.get("auto_applicable"):
                continue
            return rec
        return None

    def suggested(self, parameter: str, current: Any = None) -> Any:
        """Return the auto-applicable suggested value for ``parameter``.

        Numeric suggestions are parsed; non-numeric or absent suggestions
        return ``current`` unchanged, so this is always safe to call inline.
        """
        for rec in self.recommendations():
            if not rec.get("auto_applicable"):
                continue
            if rec.get("parameter") != parameter:
                continue
            value = rec.get("suggested")
            if value is None or value == "":
                return current
            try:
                return int(value)
            except (TypeError, ValueError):
                try:
                    return float(value)
                except (TypeError, ValueError):
                    return current
        return current
