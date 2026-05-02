"""
Synthetic historical weather generator.

When real NOAA CDO data is unavailable we build a statistically realistic
2-year hourly time series for any (lat, lon) point using:
  - Climatological temperature normals (latitude + elevation gradient)
  - Annual + semi-annual + diurnal sinusoidal cycles
  - Jet-stream pressure oscillation (20-60 day period)
  - Correlated Gaussian noise (spatially coherent weather variability)
  - Precipitation modelled as a Gamma process gated by humidity threshold

The resulting arrays are used directly by analysis/fourier.py.
"""

import numpy as np
from config import SYNTHETIC_HISTORY_DAYS

# ---------------------------------------------------------------------------
# Climatological normals by latitude (very approximate continental US)
# ---------------------------------------------------------------------------

def _mean_temp_c(lat, lon, doy):
    """Annual mean temperature as a function of lat/lon and day-of-year."""
    # Calibrated against NOAA climate normals for continental US cities.
    # Annual mean lapse: ~-0.82 C per degree north (Miami ~22.8 C, Minneapolis ~7.2 C).
    base = 44.0 - 0.82 * lat
    # Pacific coast: ocean keeps winters warmer, summers cooler
    if lon < -115:
        base += max(0, (lon + 125) * 1.0)
    # Annual cycle amplitude: grows with latitude (larger swing in north)
    amp = 6.0 + 0.6 * (lat - 25.0)
    # Peak at ~day 200 (mid-July), trough ~day 17 (mid-January)
    return base + amp * np.cos(2 * np.pi * (doy - 200) / 365.25)


def _mean_pressure_hpa(lat):
    """Rough annual-mean surface pressure by latitude."""
    # subtropical high ~30 N, low ~50 N
    return 1013.25 + 4.0 * np.cos(2 * np.pi * (lat - 30) / 40)


def _prevailing_wind(lat, lon, doy):
    """
    Return prevailing (u, v) wind in m/s as a climatological estimate.
    Westerlies dominate above ~35 N, trades below; seasonal shift included.
    """
    # Westerlies strength peaks in winter
    seasonal = 1.0 + 0.4 * np.cos(2 * np.pi * (doy - 15) / 365.25)

    if lat >= 35:
        u_mean = 5.0 * seasonal   # predominantly westerly
        v_mean = -0.5             # slight southward tilt
    elif lat >= 25:
        u_mean = -3.0 * seasonal  # NE trades
        v_mean = -1.5
    else:
        u_mean = -4.0 * seasonal
        v_mean = -1.0

    # Mountain blocking: reduce eastward flow in Rockies corridor
    if -115 < lon < -100 and lat > 35:
        u_mean *= 0.5

    return u_mean, v_mean


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def generate_history(lat, lon, hours=None, seed=None, start_doy=None):
    """
    Generate synthetic hourly weather history anchored to the actual calendar.

    start_doy : fractional day-of-year the history begins (0–365).
                Defaults to (today's DOY - hours/24), so the last entry
                aligns with today's date and seasonal conditions.

    Returns dict of 1-D numpy arrays, all length `hours`:
        t          – hours since start
        temp_c     – 2-m air temperature
        pressure   – surface pressure (hPa)
        wind_u     – eastward wind component (m/s)
        wind_v     – northward wind component (m/s)
        humidity   – relative humidity (%)
        precip     – hourly precipitation (mm)
    """
    if hours is None:
        hours = SYNTHETIC_HISTORY_DAYS * 24

    if start_doy is None:
        import datetime
        doy_today = datetime.date.today().timetuple().tm_yday
        start_doy = (doy_today - hours / 24) % 365.25

    rng = np.random.default_rng(seed if seed is not None else int(abs(lat * 100 + lon)))

    t      = np.arange(hours, dtype=float)
    # Fractional day-of-year for each hour, anchored to actual calendar date
    doy_hr = ((start_doy * 24 + t) % (365.25 * 24)) / 24

    # ---- Temperature ----
    T_mean = _mean_temp_c(lat, lon, doy_hr)
    # diurnal cycle
    T_diurnal = 6.0 * np.cos(2 * np.pi * (t % 24 - 14) / 24)
    # synoptic noise (3-7 day weather systems)
    synoptic = _colored_noise(hours, tau=96, rng=rng)   # 4-day decorrelation
    T_noise  = 4.0 * synoptic
    temp_c   = T_mean + T_diurnal + T_noise

    # ---- Pressure ----
    P_mean = _mean_pressure_hpa(lat)
    # ~20-40 day oscillation (blocking, jet stream)
    jet_osc = np.sin(2 * np.pi * t / (30 * 24)) * 4.0
    P_synoptic = 5.0 * _colored_noise(hours, tau=72, rng=rng)
    pressure   = P_mean + jet_osc + P_synoptic

    # ---- Wind ----
    u_mean, v_mean = _prevailing_wind(lat, lon, doy_hr)
    u_noise = 3.0 * _colored_noise(hours, tau=72, rng=rng)
    v_noise = 2.0 * _colored_noise(hours, tau=72, rng=rng)
    # wind and pressure gradient: when pressure falls, wind increases
    dp = np.gradient(pressure)
    wind_u = u_mean + u_noise - 0.5 * dp
    wind_v = v_mean + v_noise

    # ---- Humidity ----
    # inversely related to temperature anomaly, peaks with low pressure
    T_anom = temp_c - T_mean
    H_base = 65.0 - 1.5 * (lat - 30)  # drier inland/south
    humidity = np.clip(
        H_base - 0.8 * T_anom + 8.0 * _colored_noise(hours, tau=48, rng=rng),
        10, 99
    )

    # ---- Precipitation ----
    # Gamma process: rain when humidity > threshold + low pressure
    rain_prob = np.clip((humidity - 70) / 30.0 - dp * 0.1, 0, 1)
    precip = np.where(
        rng.random(hours) < rain_prob * 0.12,
        rng.gamma(0.8, 2.5, hours),
        0.0
    )

    return {
        "t":        t,
        "temp_c":   temp_c,
        "pressure": pressure,
        "wind_u":   wind_u,
        "wind_v":   wind_v,
        "humidity": humidity,
        "precip":   precip,
    }


# ---------------------------------------------------------------------------
# Helper: red (autocorrelated) noise
# ---------------------------------------------------------------------------

def _colored_noise(n, tau, rng):
    """
    First-order autoregressive noise with decorrelation time tau steps.
    Returns zero-mean, unit-variance series.
    """
    alpha = np.exp(-1.0 / tau)
    white = rng.standard_normal(n)
    out   = np.zeros(n)
    out[0] = white[0]
    scale  = np.sqrt(1 - alpha ** 2)
    for i in range(1, n):
        out[i] = alpha * out[i - 1] + scale * white[i]
    return out
