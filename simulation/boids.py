"""
Fluid weather boids simulation.

Each boid is a moving air-parcel with:
    pos   (lat, lon)      — current position in degrees
    vel   (u, v)          — velocity in degrees/hour (east, north)
    state dict            — temp_c, pressure, wind_u, wind_v, humidity, precip

Boid physics (all forces accumulate into acceleration each step):
  1. Separation      — avoid crowding boids of the same type
  2. Alignment       — align velocity with same-type neighbors (front coherence)
  3. Cohesion        — steer toward center of same-type cluster (system cohesion)
  4. Pressure force  — parcels accelerate down the pressure gradient
  5. Coriolis        — right-turning deflection in Northern Hemisphere
  6. Terrain block   — orographic lifting / deflection
  7. FFT attractor   — nudge state toward FFT backbone prediction
  8. Boundary        — re-inject boids that exit the domain at the opposite edge

Weather types (governs boid grouping behaviour):
  LOW   — cyclonic; high cohesion, strong alignment
  HIGH  — anticyclonic; strong separation, moderate alignment
  FRONT_COLD — fast, sharp; high alignment, moderate separation
  FRONT_WARM — slower, diffuse; low cohesion
  CLEAR — background clear-sky parcels; weakest interactions
"""

import math
import numpy as np
from config import (
    LAT_MIN, LAT_MAX, LON_MIN, LON_MAX,
    NUM_BOIDS,
    BOID_SEPARATION_RADIUS, BOID_ALIGNMENT_RADIUS, BOID_COHESION_RADIUS,
    SEPARATION_WEIGHT, ALIGNMENT_WEIGHT, COHESION_WEIGHT,
    PRESSURE_GRADIENT_WEIGHT, CORIOLIS_WEIGHT, FFT_ATTRACTOR_WEIGHT,
    TERRAIN_WEIGHT, TERRAIN_FEATURES,
    GEOSTROPHIC_WEIGHT, LATENT_HEAT_WEIGHT,
    MAX_SPEED, MIN_SPEED, DAMPING, SIM_DT_HOURS,
)

# Physical constants
_OMEGA   = 7.2921e-5   # Earth's rotation rate [rad/s]
_RHO_AIR = 1.225       # surface air density [kg/m³]
_M_PER_DEG = 111_000.0 # metres per degree of latitude

# ---- Weather type constants ----
LOW        = 0
HIGH       = 1
FRONT_COLD = 2
FRONT_WARM = 3
CLEAR      = 4

TYPE_NAMES  = {LOW: "Low", HIGH: "High", FRONT_COLD: "Cold Front",
               FRONT_WARM: "Warm Front", CLEAR: "Clear"}
TYPE_COLORS = {LOW: "#6644aa", HIGH: "#dd8844", FRONT_COLD: "#3366cc",
               FRONT_WARM: "#cc4444", CLEAR: "#88bbdd"}

# Per-type weight overrides (multipliers on global weights)
_TYPE_W = {
    #             sep  ali  coh
    LOW:        (0.7, 1.4, 1.8),
    HIGH:       (1.6, 0.9, 0.5),
    FRONT_COLD: (1.0, 1.8, 1.0),
    FRONT_WARM: (0.8, 1.1, 0.7),
    CLEAR:      (0.5, 0.6, 0.3),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _deg2rad(d): return d * math.pi / 180.0

def _haversine_deg(lat1, lon1, lat2, lon2):
    """Approximate distance in degrees (equirectangular, good enough for US)."""
    dlat = lat2 - lat1
    dlon = (lon2 - lon1) * math.cos(_deg2rad((lat1 + lat2) * 0.5))
    return math.hypot(dlat, dlon)

def _classify_type(pressure, temp_c, wind_speed):
    """Classify a parcel's weather type from its state."""
    if pressure < 1000:
        return LOW
    if pressure > 1020:
        return HIGH
    if wind_speed > 8 and temp_c < 5:
        return FRONT_COLD
    if wind_speed > 5 and temp_c > 10:
        return FRONT_WARM
    return CLEAR

def _terrain_force(lat, lon):
    """
    Orographic blocking force: parcels are pushed away from mountain centres.
    Returns (du_lat, du_lon) acceleration in deg/hr².
    """
    dlat, dlon = 0.0, 0.0
    for (clat, clon, height, sigma) in TERRAIN_FEATURES:
        d2 = (lat - clat) ** 2 + (lon - clon) ** 2
        gauss = height * math.exp(-d2 / (2 * sigma ** 2))
        # gradient of Gaussian (points away from peak)
        dlat += gauss * (lat - clat) / (sigma ** 2)
        dlon += gauss * (lon - clon) / (sigma ** 2)
    return dlat * 0.01, dlon * 0.01   # scale to reasonable acceleration


# ---------------------------------------------------------------------------
# Single boid
# ---------------------------------------------------------------------------

class WeatherBoid:
    __slots__ = (
        "lat", "lon", "vel_u", "vel_v",
        "temp_c", "pressure", "humidity", "precip",
        "wind_u", "wind_v",
        "btype", "age",
        "trail",
    )

    def __init__(self, lat, lon, vel_u, vel_v, temp_c, pressure,
                 humidity, precip, wind_u=0.0, wind_v=0.0):
        self.lat      = lat
        self.lon      = lon
        self.vel_u    = vel_u    # deg/hr eastward
        self.vel_v    = vel_v    # deg/hr northward
        self.temp_c   = temp_c
        self.pressure = pressure
        self.humidity = humidity
        self.precip   = precip
        self.wind_u   = wind_u
        self.wind_v   = wind_v
        spd = math.hypot(wind_u, wind_v)
        self.btype = _classify_type(pressure, temp_c, spd)
        self.age   = 0
        self.trail = []  # list of (lat, lon)

    @property
    def speed(self):
        return math.hypot(self.vel_u, self.vel_v)

    def record_trail(self, max_len):
        self.trail.append((self.lat, self.lon))
        if len(self.trail) > max_len:
            self.trail.pop(0)


# ---------------------------------------------------------------------------
# Boid swarm
# ---------------------------------------------------------------------------

class WeatherSwarm:
    """
    Manages the full population of weather boids and advances the simulation.
    """

    def __init__(self, rng=None):
        self.boids      = []
        self.fft_analyzers = []   # one per boid, set externally
        self.hour_elapsed  = 0.0
        self.rng = rng if rng is not None else np.random.default_rng(42)

    # ------------------------------------------------------------------
    # Initialisation from seed observations
    # ------------------------------------------------------------------

    def seed_from_observations(self, observations, n_total=NUM_BOIDS):
        """
        Seed the swarm using real NOAA observations plus synthetic fill-in.
        observations : list of dicts from data.noaa_client.fetch_station_observations
        n_total      : target boid count
        """
        self.boids = []
        for obs in observations:
            if obs["lat"] is None or obs["lon"] is None:
                continue
            if not (LAT_MIN <= obs["lat"] <= LAT_MAX and LON_MIN <= obs["lon"] <= LON_MAX):
                continue
            # wind in m/s → deg/hr  (1 deg lat ≈ 111 km; 1 m/s ≈ 0.0324 deg/hr)
            scale = 3600.0 / 111_000.0
            b = WeatherBoid(
                lat      = obs["lat"],
                lon      = obs["lon"],
                vel_u    = obs["wind_u"] * scale,
                vel_v    = obs["wind_v"] * scale,
                temp_c   = obs["temp_c"],
                pressure = obs["pressure"],
                humidity = obs["humidity"],
                precip   = obs["precip"],
                wind_u   = obs["wind_u"],
                wind_v   = obs["wind_v"],
            )
            self.boids.append(b)

        # fill remainder with synthetic boids spread across the domain
        n_fill = max(0, n_total - len(self.boids))
        self._fill_random(n_fill)

    def seed_synthetic(self, n=NUM_BOIDS):
        """Seed entirely from synthetic climatology (no API needed)."""
        self.boids = []
        self._fill_random(n)

    def apply_observations(self, observations):
        """
        Inject real NOAA station observations into the nearest boids.
        Safe to call from a background thread — only writes scalar attributes
        (protected by Python's GIL).  Called once after async NOAA fetch.
        """
        scale = 3600.0 / 111_000.0  # m/s → deg/hr
        lats  = [b.lat for b in self.boids]
        lons  = [b.lon for b in self.boids]

        for obs in observations:
            if obs.get("lat") is None or obs.get("lon") is None:
                continue
            if not (LAT_MIN <= obs["lat"] <= LAT_MAX and LON_MIN <= obs["lon"] <= LON_MAX):
                continue
            # find nearest boid
            best_i, best_d2 = 0, float("inf")
            for i, (blat, blon) in enumerate(zip(lats, lons)):
                d2 = (blat - obs["lat"]) ** 2 + (blon - obs["lon"]) ** 2
                if d2 < best_d2:
                    best_d2, best_i = d2, i
            if best_d2 > 8 ** 2:   # skip if no boid within 8 degrees
                continue
            b = self.boids[best_i]
            b.temp_c   = obs["temp_c"]
            b.pressure = obs["pressure"]
            b.humidity = obs["humidity"]
            b.precip   = obs["precip"]
            b.wind_u   = obs["wind_u"]
            b.wind_v   = obs["wind_v"]
            b.vel_u    = obs["wind_u"] * scale
            b.vel_v    = obs["wind_v"] * scale
            b.btype    = _classify_type(b.pressure, b.temp_c,
                                         math.hypot(b.wind_u, b.wind_v))

    def _fill_random(self, n):
        from data.synthetic import generate_history
        for _ in range(n):
            lat = float(self.rng.uniform(LAT_MIN + 1, LAT_MAX - 1))
            lon = float(self.rng.uniform(LON_MIN + 1, LON_MAX - 1))
            h   = generate_history(lat, lon, hours=24 * 7, seed=None)
            # use last entry as current state
            temp_c   = float(h["temp_c"][-1])
            pressure = float(h["pressure"][-1])
            humidity = float(h["humidity"][-1])
            precip   = float(h["precip"][-1])
            wu       = float(h["wind_u"][-1])
            wv       = float(h["wind_v"][-1])
            scale    = 3600.0 / 111_000.0
            b = WeatherBoid(
                lat=lat, lon=lon,
                vel_u=wu * scale, vel_v=wv * scale,
                temp_c=temp_c, pressure=pressure,
                humidity=humidity, precip=precip,
                wind_u=wu, wind_v=wv,
            )
            self.boids.append(b)

    # ------------------------------------------------------------------
    # Simulation step — fully vectorised
    # ------------------------------------------------------------------

    def step(self, dt=SIM_DT_HOURS, trail_len=25):
        """Advance simulation one step using numpy-vectorised physics."""
        n = len(self.boids)

        # ---- Extract state into contiguous arrays (fast) ----
        lats  = np.array([b.lat      for b in self.boids])
        lons  = np.array([b.lon      for b in self.boids])
        vus   = np.array([b.vel_u    for b in self.boids])
        vvs   = np.array([b.vel_v    for b in self.boids])
        prs   = np.array([b.pressure for b in self.boids])
        types = np.array([b.btype    for b in self.boids], dtype=np.int8)

        cos_lats = np.cos(np.radians(lats))

        # ---- (N, N) pairwise displacement matrices ([i,j] = j − i) ----
        DLAT = lats[np.newaxis, :] - lats[:, np.newaxis]
        DLON = (lons[np.newaxis, :] - lons[:, np.newaxis]) * cos_lats[:, np.newaxis]
        DIST = np.hypot(DLAT, DLON)
        np.fill_diagonal(DIST, np.inf)

        # Per-boid weight multipliers from weather type
        tw    = np.array([_TYPE_W.get(int(t), (1.0, 1.0, 1.0)) for t in types])
        sep_w = SEPARATION_WEIGHT * tw[:, 0]
        ali_w = ALIGNMENT_WEIGHT  * tw[:, 1]
        coh_w = COHESION_WEIGHT   * tw[:, 2]

        SAME = (types[:, np.newaxis] == types[np.newaxis, :])  # (N, N) bool

        # ---- Separation ----
        # errstate: np.where evaluates both branches; diagonal DIST=inf produces
        # inf/inf=NaN in the false branch which is then discarded — suppress it.
        SEP = DIST < BOID_SEPARATION_RADIUS
        with np.errstate(invalid="ignore", divide="ignore"):
            w_sep = np.where(SEP, (BOID_SEPARATION_RADIUS - DIST) / (DIST + 1e-6), 0.0)
        cnt   = SEP.sum(axis=1) + 1
        au    = -sep_w * (w_sep * DLON).sum(axis=1) / cnt
        av    = -sep_w * (w_sep * DLAT).sum(axis=1) / cnt

        # ---- Alignment (same-type neighbours) ----
        ALI   = SAME & (DIST < BOID_ALIGNMENT_RADIUS)
        cnt_a = ALI.sum(axis=1).clip(1)
        au   += ali_w * ((ALI * vus[np.newaxis, :]).sum(axis=1) / cnt_a - vus) * 0.5
        av   += ali_w * ((ALI * vvs[np.newaxis, :]).sum(axis=1) / cnt_a - vvs) * 0.5

        # ---- Cohesion (same-type neighbours) ----
        COH   = SAME & (DIST < BOID_COHESION_RADIUS)
        cnt_c = COH.sum(axis=1).clip(1)
        au   += coh_w * ((COH * lons[np.newaxis, :]).sum(axis=1) / cnt_c - lons) * 0.1
        av   += coh_w * ((COH * lats[np.newaxis, :]).sum(axis=1) / cnt_c - lats) * 0.1

        # ---- Pressure gradient (weighted least-squares per boid) [hPa/deg] ----
        PG     = DIST < BOID_ALIGNMENT_RADIUS
        dp_lon = ((PG * (prs[np.newaxis, :] - prs[:, np.newaxis]) * DLON).sum(axis=1)
                  / ((PG * DLON ** 2).sum(axis=1) + 1.0))
        dp_lat = ((PG * (prs[np.newaxis, :] - prs[:, np.newaxis]) * DLAT).sum(axis=1)
                  / ((PG * DLAT ** 2).sum(axis=1) + 1.0))
        au += PRESSURE_GRADIENT_WEIGHT * dp_lon * 0.005
        av += PRESSURE_GRADIENT_WEIGHT * dp_lat * 0.005

        # ---- Geostrophic wind (physically-derived balance force) ----
        # Converts estimated pressure gradient [hPa/deg] → SI [Pa/m], then
        # computes u_geo, v_geo [m/s] via geostrophic relation, converts to
        # [deg/hr], and nudges boid velocity toward that balance state.
        f_SI = 2.0 * _OMEGA * np.sin(np.radians(lats))              # [rad/s]
        f_SI = np.where(np.abs(f_SI) < 5e-5,
                        np.sign(f_SI + 1e-30) * 5e-5, f_SI)         # clamp near equator
        m_per_deg_lon = _M_PER_DEG * cos_lats
        dp_dx_SI = dp_lon * 100.0 / (m_per_deg_lon + 1e-3)          # [Pa/m]
        dp_dy_SI = dp_lat * 100.0 / _M_PER_DEG                      # [Pa/m]
        recip_fρ = 1.0 / (f_SI * _RHO_AIR)
        u_geo_ms = -recip_fρ * dp_dy_SI                              # [m/s]
        v_geo_ms =  recip_fρ * dp_dx_SI                              # [m/s]
        scale_ms = 3600.0 / _M_PER_DEG                               # m/s → deg/hr
        u_geo = np.clip(u_geo_ms * scale_ms, -MAX_SPEED, MAX_SPEED)
        v_geo = np.clip(v_geo_ms * scale_ms, -MAX_SPEED, MAX_SPEED)
        au += GEOSTROPHIC_WEIGHT * (u_geo - vus)
        av += GEOSTROPHIC_WEIGHT * (v_geo - vvs)

        # ---- Coriolis (NH right-turn) ----
        f_hr = 2 * _OMEGA * np.sin(np.radians(lats)) * 3600 * 0.01
        au  += CORIOLIS_WEIGHT *  f_hr * vvs
        av  -= CORIOLIS_WEIGHT *  f_hr * vus

        # ---- Terrain blocking (Gaussian bumps per mountain range) ----
        for (clat, clon, height, sigma) in TERRAIN_FEATURES:
            d2  = (lats - clat) ** 2 + (lons - clon) ** 2
            g   = height * np.exp(-d2 / (2.0 * sigma ** 2))
            au += TERRAIN_WEIGHT * g * (lons - clon) / (sigma ** 2) * 0.01
            av += TERRAIN_WEIGHT * g * (lats - clat) / (sigma ** 2) * 0.01

        # ---- Update velocities ----
        new_vu = (vus + au * dt) * DAMPING
        new_vv = (vvs + av * dt) * DAMPING
        spd    = np.hypot(new_vu, new_vv).clip(1e-9)
        new_vu = np.where(spd > MAX_SPEED,  new_vu * MAX_SPEED  / spd, new_vu)
        new_vv = np.where(spd > MAX_SPEED,  new_vv * MAX_SPEED  / spd, new_vv)
        new_vu = np.where(spd < MIN_SPEED,  new_vu * MIN_SPEED  / spd, new_vu)
        new_vv = np.where(spd < MIN_SPEED,  new_vv * MIN_SPEED  / spd, new_vv)

        new_lats = lats + new_vv * dt
        new_lons = lons + new_vu * dt

        # ---- Latent heat: precipitation warms parcel, drops pressure ----
        precip_arr = np.array([b.precip for b in self.boids])
        latent     = LATENT_HEAT_WEIGHT * np.maximum(0.0, precip_arr - 0.1) * dt

        # ---- Write back + per-boid ops (trail, FFT attractor, boundary) ----
        for i, b in enumerate(self.boids):
            b.vel_u = float(new_vu[i])
            b.vel_v = float(new_vv[i])
            b.lat   = float(new_lats[i])
            b.lon   = float(new_lons[i])

            # Latent heat: condensation warms air and lowers surface pressure
            if latent[i] > 0.0:
                b.temp_c   += float(latent[i])
                b.pressure -= float(latent[i]) * 0.4   # ~0.4 hPa / °C warming

            if self.fft_analyzers and i < len(self.fft_analyzers):
                self._apply_fft_attractor(b, i, dt)

            b.btype = _classify_type(b.pressure, b.temp_c,
                                      math.hypot(b.wind_u, b.wind_v))
            b.age  += 1
            b.record_trail(trail_len)

            if self._is_out_of_bounds(b):
                self.boids[i] = self._respawn(b)

        self.hour_elapsed += dt

    def _apply_fft_attractor(self, b, i, dt):
        """Nudge boid state toward the FFT-predicted attractor."""
        analyzer = self.fft_analyzers[i]
        pred     = analyzer.predict(self.hour_elapsed)
        alpha    = FFT_ATTRACTOR_WEIGHT * dt

        b.temp_c   += alpha * (pred.get("temp_c",   b.temp_c)   - b.temp_c)
        b.pressure += alpha * (pred.get("pressure", b.pressure) - b.pressure)
        b.humidity += alpha * (pred.get("humidity", b.humidity) - b.humidity)
        b.precip    = max(0, b.precip + alpha * (pred.get("precip", 0) - b.precip))

        # also nudge velocity toward FFT-predicted wind
        wu_pred = pred.get("wind_u", b.wind_u)
        wv_pred = pred.get("wind_v", b.wind_v)
        scale   = 3600.0 / 111_000.0
        b.vel_u += alpha * (wu_pred * scale - b.vel_u)
        b.vel_v += alpha * (wv_pred * scale - b.vel_v)
        b.wind_u = b.vel_u / scale
        b.wind_v = b.vel_v / scale

    # ------------------------------------------------------------------
    # Boundary handling
    # ------------------------------------------------------------------

    def _is_out_of_bounds(self, b):
        return not (LAT_MIN <= b.lat <= LAT_MAX and LON_MIN <= b.lon <= LON_MAX)

    def _respawn(self, b):
        """Re-enter boid from the opposite edge, inheriting type / state."""
        from data.synthetic import generate_history

        # pick a random edge entry point
        edge = int(self.rng.integers(0, 4))
        if edge == 0:   # enter from west
            lat = b.lat if LAT_MIN <= b.lat <= LAT_MAX else float(self.rng.uniform(LAT_MIN, LAT_MAX))
            lon = LON_MIN + 0.5
        elif edge == 1: # enter from east
            lat = b.lat if LAT_MIN <= b.lat <= LAT_MAX else float(self.rng.uniform(LAT_MIN, LAT_MAX))
            lon = LON_MAX - 0.5
        elif edge == 2: # enter from south
            lat = LAT_MIN + 0.5
            lon = float(self.rng.uniform(LON_MIN, LON_MAX))
        else:           # enter from north
            lat = LAT_MAX - 0.5
            lon = float(self.rng.uniform(LON_MIN, LON_MAX))

        h = generate_history(lat, lon, hours=24, seed=None)
        b.lat      = lat
        b.lon      = lon
        b.temp_c   = float(h["temp_c"][-1])
        b.pressure = float(h["pressure"][-1])
        b.humidity = float(h["humidity"][-1])
        b.precip   = float(h["precip"][-1])
        b.wind_u   = float(h["wind_u"][-1])
        b.wind_v   = float(h["wind_v"][-1])
        scale = 3600.0 / 111_000.0
        b.vel_u = b.wind_u * scale
        b.vel_v = b.wind_v * scale
        b.btype = _classify_type(b.pressure, b.temp_c,
                                  math.hypot(b.wind_u, b.wind_v))
        b.trail = []
        return b

    # ------------------------------------------------------------------
    # Snapshot for rendering
    # ------------------------------------------------------------------

    def snapshot(self):
        """Return arrays suitable for fast rendering."""
        n = len(self.boids)
        lats      = np.empty(n); lons      = np.empty(n)
        vus       = np.empty(n); vvs       = np.empty(n)
        temps     = np.empty(n); pressures = np.empty(n)
        humids    = np.empty(n); precips   = np.empty(n)
        btypes    = np.empty(n, dtype=int)
        for i, b in enumerate(self.boids):
            lats[i]      = b.lat;      lons[i]      = b.lon
            vus[i]       = b.vel_u;    vvs[i]       = b.vel_v
            temps[i]     = b.temp_c;   pressures[i] = b.pressure
            humids[i]    = b.humidity; precips[i]   = b.precip
            btypes[i]    = b.btype
        return {
            "lats": lats, "lons": lons,
            "vel_u": vus, "vel_v": vvs,
            "temp_c": temps, "pressure": pressures,
            "humidity": humids, "precip": precips,
            "btypes": btypes,
            "trails": [list(b.trail) for b in self.boids],
        }

    # ------------------------------------------------------------------
    # Timeline construction (past recordings + fast-forward future)
    # ------------------------------------------------------------------

    def build_timeline(self, recorded_snapshots=None,
                       future_hours=120.0,
                       snapshot_interval_hours=1.0,
                       dt=None):
        """
        Build an ordered list of {hour_offset, snap} covering:
          • Recorded past  — real snapshots loaded from disk (hour_offset < 0)
          • Present        — current swarm state (hour_offset = 0)
          • Predicted future — fast-forward of a saved/restored swarm clone

        Returns list sorted by hour_offset.  Does NOT modify swarm state.
        """
        import logging
        log = logging.getLogger(__name__)

        if dt is None:
            dt = SIM_DT_HOURS

        result = []

        # Past: from recorder (hour_offset already negative)
        if recorded_snapshots:
            for entry in recorded_snapshots:
                result.append({
                    "hour_offset": entry["hour_offset"],
                    "snap":        entry["snap"],
                })
            log.info("Timeline: loaded %d past snapshots", len(recorded_snapshots))

        # Present
        result.append({"hour_offset": 0.0, "snap": self.snapshot()})

        # Future: run a saved-state clone forward
        log.info("Timeline: fast-forwarding %.0f hours into future …", future_hours)
        future = self._fast_forward(future_hours, snapshot_interval_hours, dt)
        result.extend(future)
        log.info("Timeline: built %d total entries (past+present+future)", len(result))

        result.sort(key=lambda x: x["hour_offset"])
        return result

    def _fast_forward(self, future_hours, interval_hours, dt):
        """
        Temporarily run the swarm forward, collecting snapshots, then restore
        original state.  Returns list of {hour_offset, snap}.
        """
        # --- Save full boid state ---
        saved = [
            (b.lat, b.lon, b.vel_u, b.vel_v,
             b.temp_c, b.pressure, b.humidity, b.precip,
             b.wind_u, b.wind_v, b.btype, b.age, list(b.trail))
            for b in self.boids
        ]
        saved_hour = self.hour_elapsed

        snaps        = []
        steps_per_snap = max(1, round(interval_hours / dt))
        total_steps    = max(1, round(future_hours  / dt))

        try:
            for step_i in range(1, total_steps + 1):
                self.step(dt=dt, trail_len=10)
                if step_i % steps_per_snap == 0:
                    snaps.append({
                        "hour_offset": float(step_i * dt),
                        "snap":        self.snapshot(),
                    })
        finally:
            # --- Restore unconditionally ---
            self.hour_elapsed = saved_hour
            for i, b in enumerate(self.boids):
                s = saved[i]
                (b.lat, b.lon, b.vel_u, b.vel_v,
                 b.temp_c, b.pressure, b.humidity, b.precip,
                 b.wind_u, b.wind_v, b.btype, b.age) = s[:12]
                b.trail = s[12]

        return snaps
