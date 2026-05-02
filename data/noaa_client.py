"""
NOAA Weather API client.
Uses api.weather.gov (no key required) to seed boids with current conditions.
Falls back gracefully if the API is unavailable or rate-limited.
"""

import math
import time
import logging
import requests

log = logging.getLogger(__name__)

_SESSION = None


def _session():
    global _SESSION
    if _SESSION is None:
        _SESSION = requests.Session()
        _SESSION.headers.update({
            "User-Agent": "NOAAAI-WeatherViz/1.0 (educational; contact=user@example.com)",
            "Accept": "application/geo+json,application/json",
        })
    return _SESSION


def _get(url, timeout=6):
    try:
        r = _session().get(url, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        log.debug("NOAA request failed %s: %s", url, exc)
        return None


# ---------------------------------------------------------------------------
# Point metadata
# ---------------------------------------------------------------------------

def get_point_meta(lat, lon):
    """Return gridpoint metadata dict or None."""
    return _get(f"https://api.weather.gov/points/{lat:.4f},{lon:.4f}")


# ---------------------------------------------------------------------------
# Latest station observation → normalised dict
# ---------------------------------------------------------------------------

def _extract(props, key):
    v = props.get(key) or {}
    val = v.get("value")
    return float(val) if val is not None else None


def get_station_observation(station_id):
    """Return normalised observation dict for an ASOS station id."""
    data = _get(f"https://api.weather.gov/stations/{station_id}/observations/latest")
    if not data:
        return None
    props = data.get("properties", {})

    temp_c   = _extract(props, "temperature")
    wind_spd = _extract(props, "windSpeed")      # m/s
    wind_dir = _extract(props, "windDirection")  # degrees from N
    pressure = _extract(props, "barometricPressure")  # Pa → hPa below
    humidity = _extract(props, "relativeHumidity")
    precip   = _extract(props, "precipitationLastHour")  # mm

    if temp_c is None:
        return None

    # Convert wind from polar to Cartesian components (met convention: dir FROM)
    u, v = 0.0, 0.0
    if wind_spd is not None and wind_dir is not None:
        rad = math.radians(wind_dir)
        # meteorological: wind FROM direction, so velocity is opposite
        u = -wind_spd * math.sin(rad)
        v = -wind_spd * math.cos(rad)

    # Extract lat/lon from geometry
    geo = data.get("geometry") or {}
    coords = geo.get("coordinates", [None, None])
    lat = coords[1] if len(coords) > 1 else None
    lon = coords[0] if len(coords) > 0 else None

    return {
        "lat":      lat,
        "lon":      lon,
        "temp_c":   temp_c,
        "wind_u":   u,
        "wind_v":   v,
        "pressure": (pressure / 100.0) if pressure else 1013.25,  # hPa
        "humidity": humidity if humidity is not None else 60.0,
        "precip":   precip  if precip   is not None else 0.0,
        "station":  station_id,
    }


# ---------------------------------------------------------------------------
# Bulk fetch — parallel with ThreadPoolExecutor
# ---------------------------------------------------------------------------

def fetch_station_observations_parallel(station_ids, max_workers=8):
    """
    Fetch observations for all station IDs in parallel.
    max_workers=8 matches urllib3's default connection pool size (10) with
    headroom, avoiding the 'Connection pool is full' warning.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FutTimeout

    def _fetch_one(sid):
        try:
            return get_station_observation(sid)
        except Exception:
            return None

    results = []
    # Generous timeout: NOAA can be slow but each request already has a 6s
    # socket timeout, so worst case = ceil(n / workers) * 6s + slack.
    wall_timeout = (len(station_ids) / max_workers + 1) * 8

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_fetch_one, sid): sid for sid in station_ids}
        try:
            for fut in as_completed(futures, timeout=wall_timeout):
                try:
                    obs = fut.result()
                    if obs:
                        results.append(obs)
                except Exception:
                    pass
        except FutTimeout:
            # Collect whatever managed to finish before the wall clock expired
            for fut in futures:
                if fut.done():
                    try:
                        obs = fut.result()
                        if obs:
                            results.append(obs)
                    except Exception:
                        pass
            log.info("NOAA fetch timed out; collected %d partial results", len(results))
    return results


def fetch_station_observations(station_ids, delay=0.05):
    """Sequential fetch (kept for compatibility). Prefer parallel version."""
    results = []
    for sid in station_ids:
        obs = get_station_observation(sid)
        if obs:
            results.append(obs)
        time.sleep(delay)
    return results


# ---------------------------------------------------------------------------
# Hourly forecast for a grid point (for FFT seeding)
# ---------------------------------------------------------------------------

def get_hourly_forecast(lat, lon):
    """
    Return list of hourly forecast dicts for the next ~156 hours.
    Each dict: {hour_offset, temp_c, wind_u, wind_v, precip_prob, humidity}
    """
    meta = get_point_meta(lat, lon)
    if not meta:
        return []

    props     = meta.get("properties", {})
    grid_id   = props.get("gridId")
    grid_x    = props.get("gridX")
    grid_y    = props.get("gridY")
    if not all([grid_id, grid_x, grid_y]):
        return []

    data = _get(
        f"https://api.weather.gov/gridpoints/{grid_id}/{grid_x},{grid_y}/forecast/hourly"
    )
    if not data:
        return []

    periods = data.get("properties", {}).get("periods", [])
    results = []
    for i, p in enumerate(periods):
        temp_f    = p.get("temperature", 70)
        temp_c    = (temp_f - 32) * 5 / 9
        wind_str  = p.get("windSpeed", "0 mph")
        wind_dir  = p.get("windDirection", "N")
        precip    = p.get("probabilityOfPrecipitation", {}).get("value", 0) or 0
        humidity  = p.get("relativeHumidity", {}).get("value", 60) or 60

        spd_mph = float(wind_str.split()[0]) if wind_str else 0.0
        spd_ms  = spd_mph * 0.44704
        dir_deg = _dir_to_deg(wind_dir)
        rad     = math.radians(dir_deg)
        u       = -spd_ms * math.sin(rad)
        v       = -spd_ms * math.cos(rad)

        results.append({
            "hour_offset": i,
            "temp_c":      temp_c,
            "wind_u":      u,
            "wind_v":      v,
            "precip_prob": precip / 100.0,
            "humidity":    float(humidity),
        })
    return results


def _dir_to_deg(d):
    mapping = {
        "N": 0, "NNE": 22.5, "NE": 45, "ENE": 67.5,
        "E": 90, "ESE": 112.5, "SE": 135, "SSE": 157.5,
        "S": 180, "SSW": 202.5, "SW": 225, "WSW": 247.5,
        "W": 270, "WNW": 292.5, "NW": 315, "NNW": 337.5,
    }
    return mapping.get(d, 0)
