"""
Fourier (FFT) analysis and prediction engine.

For each weather variable at each boid's location we:
  1. Compute the FFT of the historical time series.
  2. Keep the DOMINANT_MODES largest-amplitude frequency components.
  3. Reconstruct a smooth 'backbone' signal (the deterministic part of weather).
  4. Extrapolate that backbone arbitrarily far into the future by advancing phases.

The 2-D spatial FFT of the pressure field is also used to identify dominant
mesoscale wave patterns (ridges / troughs) — these feed the boid cohesion and
separation forces at each animation frame.

Usage
-----
    analyzer = WeatherFFT(history_dict)  # history from data/synthetic.py
    state_now = analyzer.predict(hour_offset=0)   # current backbone
    state_24h = analyzer.predict(hour_offset=24)  # 24-hour forecast backbone
"""

import numpy as np
from config import DOMINANT_MODES, PREDICTION_HOURS

VARIABLES = ("temp_c", "pressure", "wind_u", "wind_v", "humidity", "precip")


class WeatherFFT:
    """
    Per-location FFT analyser.

    Parameters
    ----------
    history : dict
        Output of data.synthetic.generate_history().
    dt_hours : float
        Sampling interval of the history (default 1 h).
    """

    def __init__(self, history, dt_hours=1.0):
        self.dt = dt_hours
        self.n  = len(history["t"])
        self._means    = {}
        self._spectra  = {}   # {var: (freqs, amplitudes, phases)}

        for var in VARIABLES:
            if var not in history:
                continue
            series = np.asarray(history[var], dtype=float)
            self._means[var] = series.mean()
            self._spectra[var] = self._analyse(series - self._means[var])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, hour_offset):
        """
        Return predicted weather state dict at `hour_offset` hours from
        the end of the history record.
        """
        out = {}
        for var, (freqs, amps, phases) in self._spectra.items():
            out[var] = self._means[var] + self._reconstruct(freqs, amps, phases, hour_offset)
        # clip physically unreasonable values
        if "humidity" in out:
            out["humidity"] = float(np.clip(out["humidity"], 10, 99))
        if "precip" in out:
            out["precip"] = float(max(0.0, out["precip"]))
        return out

    def dominant_period_hours(self, var):
        """Return the period (in hours) of the strongest frequency component."""
        if var not in self._spectra:
            return None
        freqs, amps, _ = self._spectra[var]
        idx = np.argmax(amps)
        f   = freqs[idx]
        return (1.0 / f) if f > 0 else None

    def spectral_energy(self, var):
        """Total spectral energy kept in the dominant modes."""
        if var not in self._spectra:
            return 0.0
        _, amps, _ = self._spectra[var]
        return float(np.sum(amps ** 2))

    # ------------------------------------------------------------------
    # Internal FFT mechanics
    # ------------------------------------------------------------------

    def _analyse(self, series):
        """
        FFT a zero-mean series; return (freqs, amplitudes, phases) of the
        DOMINANT_MODES strongest positive-frequency components.
        """
        N   = len(series)
        fft = np.fft.rfft(series)
        freqs_all = np.fft.rfftfreq(N, d=self.dt)  # cycles per hour

        amps_all   = np.abs(fft) / N
        phases_all = np.angle(fft)

        # exclude DC (index 0)
        valid = np.arange(1, len(freqs_all))
        top_k = np.argsort(amps_all[valid])[-DOMINANT_MODES:][::-1]
        idx   = valid[top_k]

        return freqs_all[idx], amps_all[idx] * 2, phases_all[idx]

    def _reconstruct(self, freqs, amps, phases, hour_offset):
        """
        Evaluate the dominant-mode reconstruction at a single time point
        `hour_offset` hours ahead of the end of the history.
        t_end = self.n * self.dt hours since epoch
        """
        t = self.n * self.dt + hour_offset
        return float(np.sum(amps * np.cos(2 * np.pi * freqs * t + phases)))


# ---------------------------------------------------------------------------
# Spatial FFT — pressure field wave analysis
# ---------------------------------------------------------------------------

class SpatialPressureFFT:
    """
    2-D FFT of the instantaneous pressure field on the boid grid.

    Identifies dominant spatial modes (ridges / troughs) whose wavelength
    and orientation feed into the boid alignment force — nearby boids
    should align in directions consistent with the dominant wave mode.
    """

    def __init__(self, grid_lats, grid_lons):
        self.lats = grid_lats  # 1-D arrays defining the regular grid
        self.lons = grid_lons
        self._last_modes = None

    def update(self, pressure_2d):
        """
        pressure_2d : 2-D array shape (nlat, nlon) of pressure values.
        Compute 2-D FFT, extract dominant modes, store gradient vectors.
        """
        P    = pressure_2d - pressure_2d.mean()
        F    = np.fft.fft2(P)
        amps = np.abs(F)

        # zero out DC
        amps[0, 0] = 0

        # find the single dominant spatial mode
        flat_idx  = np.argmax(amps)
        r, c      = np.unravel_index(flat_idx, amps.shape)
        phase_rc  = np.angle(F[r, c])

        # wave vector in grid-index space
        nlat, nlon = P.shape
        kr = r if r < nlat // 2 else r - nlat
        kc = c if c < nlon // 2 else c - nlon

        self._last_modes = {
            "kr": kr, "kc": kc, "amp": float(amps[r, c]),
            "phase": phase_rc,
        }

    def wave_gradient(self, lat, lon):
        """
        Return a unit 2-D gradient vector (u, v) in lat/lon space for the
        dominant wave mode at location (lat, lon).  Used as the boid
        alignment attractor direction in the spatial-FFT correction.
        """
        if self._last_modes is None:
            return 0.0, 0.0
        m    = self._last_modes
        dlat = (lat  - self.lats[0]) / (self.lats[-1]  - self.lats[0]  + 1e-9)
        dlon = (lon  - self.lons[0]) / (self.lons[-1]  - self.lons[0]  + 1e-9)
        phase_local = (2 * np.pi * (m["kr"] * dlat + m["kc"] * dlon) + m["phase"])
        # gradient direction perpendicular to wave fronts
        grad_lat = -m["kr"] * np.sin(phase_local)
        grad_lon =  m["kc"] * np.sin(phase_local)
        mag = np.hypot(grad_lat, grad_lon) + 1e-9
        return grad_lon / mag, grad_lat / mag   # (east, north) convention


# ---------------------------------------------------------------------------
# Batch builder — create one WeatherFFT per boid from its seeded history
# ---------------------------------------------------------------------------

def build_fft_analyzers(boid_histories):
    """
    boid_histories : list of history dicts (one per boid).
    Returns list of WeatherFFT objects.
    """
    return [WeatherFFT(h) for h in boid_histories]
