from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PROGRESS_PATH = ROOT / "data" / "progress.json"


def load_attempts(path: Path = PROGRESS_PATH) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Could not read progress file {path}: {error}") from error
    if not isinstance(payload, list) or any(not isinstance(item, dict) for item in payload):
        raise RuntimeError(f"Progress file {path} has an invalid format.")
    return payload


def save_attempt(attempt: dict[str, Any], path: Path = PROGRESS_PATH) -> None:
    if not isinstance(attempt, dict):
        raise TypeError("An attempt must be a dictionary.")
    record = dict(attempt)
    record["timestamp"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    records = load_attempts(path)
    records.append(record)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary_path.write_text(
            json.dumps(records, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(temporary_path, path)
    except OSError as error:
        raise RuntimeError(f"Could not save progress file {path}: {error}") from error
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
