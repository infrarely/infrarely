from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Iterable, Tuple

BASE_DIR = Path(os.getenv("INFRARELY_HOME", "~/.infrarely")).expanduser()

MEMORY_DB = BASE_DIR / "memory.db"
TRACES_DB = BASE_DIR / "traces.db"
STATE_DB = BASE_DIR / "state.db"

LOG_DIR = BASE_DIR / "logs"
LOG_FILE = LOG_DIR / "infrarely.log"
ERROR_LOG_DIR = LOG_DIR / "errors"
ERROR_LOG_FILE = ERROR_LOG_DIR / "error.log"
TRACE_DIR = LOG_DIR / "traces"

CACHE_DIR = BASE_DIR / "cache"
TEMP_DIR = BASE_DIR / "temp"

VERSIONS_DIR = BASE_DIR / "versions"
PACKAGES_DIR = BASE_DIR / "packages"

FAILURE_REPORTS_DIR = LOG_DIR / "failure_reports"
ANALYTICS_DIR = LOG_DIR / "analytics"


def ensure_runtime_layout() -> None:
    for directory in (
        BASE_DIR,
        LOG_DIR,
        ERROR_LOG_DIR,
        TRACE_DIR,
        CACHE_DIR,
        TEMP_DIR,
        VERSIONS_DIR,
        PACKAGES_DIR,
        FAILURE_REPORTS_DIR,
        ANALYTICS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def _legacy_mappings(cwd: Path) -> Iterable[Tuple[Path, Path]]:
    return (
        (cwd / "aos_memory.db", MEMORY_DB),
        (cwd / "aos_traces.db", TRACES_DB),
        (cwd / "aos_state.db", STATE_DB),
        (cwd / "logs" / "aos.log", LOG_FILE),
        (cwd / ".aos_versions", VERSIONS_DIR),
        (cwd / ".aos_packages", PACKAGES_DIR),
    )


def migrate_legacy_runtime_artifacts(cwd: Path | None = None) -> None:
    ensure_runtime_layout()
    source_root = cwd or Path.cwd()

    for legacy_path, target_path in _legacy_mappings(source_root):
        if not legacy_path.exists() or target_path.exists():
            continue

        target_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.move(str(legacy_path), str(target_path))
        except OSError:
            continue


ensure_runtime_layout()
migrate_legacy_runtime_artifacts()
