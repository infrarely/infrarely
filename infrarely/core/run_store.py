from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

RUNS_DIR = Path(".infrarely/runs")


class RunStore:
    """Persist and retrieve execution runs for replay and diagnostics."""

    _RUN_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{4,128}$")

    def __init__(self, runs_dir: Path = RUNS_DIR):
        self._runs_dir = Path(runs_dir)
        self._runs_dir.mkdir(parents=True, exist_ok=True)

    def _path_for(self, run_id: str) -> Path:
        if not isinstance(run_id, str) or not self._RUN_ID_RE.fullmatch(run_id):
            raise ValueError(
                "Invalid run_id. Use 4-128 chars: letters, digits, underscore, dot, colon, dash."
            )
        return self._runs_dir / f"{run_id}.json"

    def save(self, run_id: str, payload: Dict[str, Any]) -> Path:
        path = self._path_for(run_id)
        payload_to_save = dict(payload)
        payload_to_save.setdefault("run_id", run_id)
        payload_to_save["saved_at"] = datetime.now(timezone.utc).isoformat()

        # Atomic write: write temp file then replace.
        temp_name = f".{run_id}.{uuid.uuid4().hex}.tmp"
        temp_path = self._runs_dir / temp_name
        temp_path.write_text(
            json.dumps(payload_to_save, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
        temp_path.replace(path)
        return path

    def load(self, run_id: str) -> Dict[str, Any]:
        path = self._path_for(run_id)
        if not path.exists():
            raise FileNotFoundError(f"No run found: {run_id}")
        return json.loads(path.read_text(encoding="utf-8"))

    def list_runs(self, limit: int = 20) -> List[Dict[str, Any]]:
        safe_limit = max(1, min(int(limit), 500))
        files = sorted(
            self._runs_dir.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )[:safe_limit]

        runs: List[Dict[str, Any]] = []
        for file_path in files:
            try:
                data = json.loads(file_path.read_text(encoding="utf-8"))
                run_id = data.get("run_id") or file_path.stem
                input_preview = str(data.get("input", ""))[:60]
                tool_calls = data.get("tool_calls", [])
                if not isinstance(tool_calls, list):
                    tool_calls = []
                runs.append(
                    {
                        "run_id": run_id,
                        "input": input_preview,
                        "saved_at": data.get("saved_at"),
                        "tool_calls": len(tool_calls),
                        "success": bool(data.get("success", True)),
                    }
                )
            except Exception:
                # Skip malformed entries instead of failing listing.
                continue
        return runs
