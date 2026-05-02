"""
Persistent data recorder for NOAA observations and swarm snapshots.

Files (append-only JSONL, pruned to keep last KEEP_DAYS days):
  cache/observations_history.jsonl  — station observations per fetch cycle
  cache/snapshots_history.jsonl     — serialised swarm snapshots per sim step

Design notes:
  - Each line is a complete JSON object terminated with \\n; safe to append
    concurrently as long as the GIL is held during the write() call (it is).
  - Numpy arrays are serialised as lists; dtypes recovered on load.
  - Timestamps are ISO-8601 UTC strings; always use timezone-aware datetimes.
  - Pruning runs at startup (load time), not on every write.
"""

import json
import os
import datetime
import numpy as np
import logging

log = logging.getLogger(__name__)

_CACHE_DIR   = os.path.join(os.path.dirname(__file__), "..", "cache")
_OBS_FILE    = os.path.join(_CACHE_DIR, "observations_history.jsonl")
_SNAP_FILE   = os.path.join(_CACHE_DIR, "snapshots_history.jsonl")
KEEP_DAYS    = 7   # prune records older than this


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ensure_cache():
    os.makedirs(_CACHE_DIR, exist_ok=True)


def _now_utc() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


def _iso(dt: datetime.datetime) -> str:
    return dt.isoformat()


def _parse_iso(s: str) -> datetime.datetime:
    return datetime.datetime.fromisoformat(s)


def _append(path: str, obj: dict):
    _ensure_cache()
    line = json.dumps(obj, separators=(",", ":")) + "\n"
    with open(path, "a", encoding="utf-8") as f:
        f.write(line)


def _load_lines(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass   # skip corrupt lines silently
    return records


def _prune(path: str, keep_days: int = KEEP_DAYS):
    """Rewrite file keeping only lines newer than keep_days ago."""
    if not os.path.exists(path):
        return
    cutoff = _now_utc() - datetime.timedelta(days=keep_days)
    kept = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                ts = _parse_iso(obj.get("timestamp", "1970-01-01T00:00:00+00:00"))
                if ts >= cutoff:
                    kept.append(line)
            except Exception:
                pass
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(kept))
        if kept:
            f.write("\n")
    log.debug("Pruned %s: kept %d records", path, len(kept))


# ---------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------

def record_observations(observations: list[dict],
                        timestamp: datetime.datetime | None = None):
    """
    Append one fetch-cycle of station observations to disk.

    observations : list of dicts from noaa_client.get_station_observation()
    timestamp    : UTC datetime of the fetch; defaults to now
    """
    if not observations:
        return
    ts = timestamp or _now_utc()
    record = {
        "timestamp":    _iso(ts),
        "observations": observations,
    }
    _append(_OBS_FILE, record)
    log.debug("Recorded %d observations at %s", len(observations), _iso(ts))


def load_station_history(hours: float = 72.0) -> list[dict]:
    """
    Return all observation records from the last *hours* hours.
    Each record: {"timestamp": str, "observations": [...]}
    """
    _prune(_OBS_FILE)
    cutoff = _now_utc() - datetime.timedelta(hours=hours)
    records = _load_lines(_OBS_FILE)
    result  = []
    for r in records:
        try:
            ts = _parse_iso(r["timestamp"])
            if ts >= cutoff:
                result.append(r)
        except Exception:
            pass
    return result


# ---------------------------------------------------------------------------
# Snapshots
# ---------------------------------------------------------------------------

def _serialise_snap(snap: dict) -> dict:
    """Convert numpy arrays in a swarm snapshot to JSON-safe lists."""
    out = {}
    for k, v in snap.items():
        if k == "trails":
            # list of deques/lists of (lat, lon) pairs
            out[k] = [
                [[float(p[0]), float(p[1])] for p in trail]
                for trail in v
            ]
        elif isinstance(v, np.ndarray):
            out[k] = v.tolist()
        else:
            out[k] = v
    return out


def _deserialise_snap(obj: dict) -> dict:
    """Recover numpy arrays from a deserialised snapshot dict."""
    snap = {}
    array_keys = {"lats", "lons", "vel_u", "vel_v",
                  "temp_c", "pressure", "humidity", "precip", "btypes"}
    for k, v in obj.items():
        if k == "trails":
            snap[k] = [
                [tuple(p) for p in trail]
                for trail in v
            ]
        elif k in array_keys:
            snap[k] = np.array(v)
        else:
            snap[k] = v
    return snap


def record_snapshot(snap: dict, abs_utc_time: datetime.datetime | None = None):
    """
    Serialise and append a swarm snapshot to disk.

    snap         : dict from WeatherSwarm.snapshot()
    abs_utc_time : UTC wall-clock time this snapshot represents
    """
    ts = abs_utc_time or _now_utc()
    record = {
        "timestamp": _iso(ts),
        "snap":      _serialise_snap(snap),
    }
    _append(_SNAP_FILE, record)


def load_recent_snapshots(hours: float = 72.0) -> list[dict]:
    """
    Return list of {"timestamp": datetime, "hour_offset": float, "snap": dict}
    for snapshots taken in the last *hours* hours, sorted oldest-first.

    hour_offset is negative (past) relative to now.
    """
    _prune(_SNAP_FILE)
    now     = _now_utc()
    cutoff  = now - datetime.timedelta(hours=hours)
    records = _load_lines(_SNAP_FILE)
    result  = []
    for r in records:
        try:
            ts = _parse_iso(r["timestamp"])
            if ts < cutoff:
                continue
            offset = (ts - now).total_seconds() / 3600.0  # negative = past
            snap   = _deserialise_snap(r["snap"])
            result.append({
                "timestamp":   ts,
                "hour_offset": offset,
                "snap":        snap,
            })
        except Exception as exc:
            log.debug("Skipping corrupt snapshot record: %s", exc)
    result.sort(key=lambda x: x["timestamp"])
    return result


def snapshot_count() -> int:
    """Return total number of snapshots on disk (for diagnostics)."""
    if not os.path.exists(_SNAP_FILE):
        return 0
    count = 0
    with open(_SNAP_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count
