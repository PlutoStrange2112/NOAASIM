"""
NOAA AI Weather Visualization
==============================
Run:  python main.py           (fetches live NOAA data before opening window)
      python main.py --no-api  (synthetic fallback — temperatures approximate)
      python main.py --boids 500
      python main.py --full-history  (730-day FFT baseline — slower startup)

Startup sequence:
  1. Synthetic grid built (positions only — weather state ignored until step 3)
  2. FFT analysers built from calibrated synthetic baseline
  3. ** Live NOAA fetch ** (35 stations, parallel, ~10-20 s)
  4. All 350 boids seeded by IDW-interpolating the real station data
  5. Past snapshots loaded from disk
  6. Timeline fast-forwarded 120 h
  7. Window opens — temperatures match real current conditions

Subsequent runs:
  Each session records NOAA observations and live snapshots to cache/,
  which future sessions load for the historical half of the slider timeline.

Background refresh:
  After the window opens a daemon thread re-fetches NOAA every 30 minutes
  and injects updates so the live view stays current.
"""

import argparse
import logging
import threading
import datetime
import time
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)-8s %(name)s — %(message)s",
)
log = logging.getLogger("main")

parser = argparse.ArgumentParser(description="NOAA AI Weather Visualization")
parser.add_argument("--no-api",       action="store_true", help="Skip NOAA API (synthetic fallback)")
parser.add_argument("--boids",        type=int, default=None, help="Override boid count")
parser.add_argument("--full-history", action="store_true",
                    help="Use 730-day FFT baseline (slower startup, richer prediction)")
args = parser.parse_args()

NOAA_REFRESH_INTERVAL = 30 * 60   # seconds between background re-fetches


def _fetch_noaa(station_ids):
    """Fetch station observations; returns list or [] on failure."""
    try:
        from data.noaa_client import fetch_station_observations_parallel
        obs = fetch_station_observations_parallel(station_ids)
        return obs or []
    except Exception as exc:
        log.warning("NOAA fetch failed: %s", exc)
        return []


def main():
    import config
    if args.boids:
        config.NUM_BOIDS = args.boids

    rng        = np.random.default_rng(0)
    start_time = datetime.datetime.now(datetime.timezone.utc)

    # ------------------------------------------------------------------
    # 1. Build boid grid (positions only; weather state set in step 4)
    # ------------------------------------------------------------------
    log.info("Initialising %d boid positions …", config.NUM_BOIDS)
    from simulation.boids import WeatherSwarm
    swarm = WeatherSwarm(rng=rng)
    swarm.seed_synthetic(n=config.NUM_BOIDS)   # positions + placeholder weather
    log.info("  Boid positions ready")

    # ------------------------------------------------------------------
    # 2. Build FFT analysers (calibrated synthetic seasonal baseline)
    #    These drive the prediction engine; current conditions come from NOAA.
    # ------------------------------------------------------------------
    history_days = 730 if args.full_history else 30
    log.info("Building FFT analysers (%d-day seasonal baseline) …", history_days)
    from data.synthetic import generate_history
    from analysis.fourier import WeatherFFT

    history_hours = history_days * 24
    analyzers = []
    total = len(swarm.boids)
    for idx, boid in enumerate(swarm.boids):
        hist = generate_history(boid.lat, boid.lon, hours=history_hours)
        analyzers.append(WeatherFFT(hist))
        if (idx + 1) % 100 == 0 or (idx + 1) == total:
            log.info("  FFT: %d / %d", idx + 1, total)
    swarm.fft_analyzers = analyzers
    log.info("  FFT analysers ready")

    # ------------------------------------------------------------------
    # 3. Live NOAA fetch (synchronous — window waits for real data)
    # ------------------------------------------------------------------
    observations = []
    if not args.no_api:
        log.info("Fetching live NOAA observations from %d stations …",
                 len(config.SEED_STATIONS))
        observations = _fetch_noaa(config.SEED_STATIONS)
        if observations:
            log.info("  Received %d station observations", len(observations))
        else:
            log.warning("  No NOAA data returned — temperatures will be approximate")
    else:
        log.info("--no-api set; skipping NOAA fetch (synthetic temperatures only)")

    # ------------------------------------------------------------------
    # 4. Seed all boids from real NOAA data via IDW interpolation
    # ------------------------------------------------------------------
    from data import recorder

    if observations:
        ok = swarm.seed_from_real_interpolation(observations)
        if ok:
            log.info("  All %d boids seeded from real NOAA data (IDW interpolation)",
                     len(swarm.boids))
        # Record observations for future sessions' historical timeline
        try:
            recorder.record_observations(observations, timestamp=start_time)
        except Exception as exc:
            log.warning("Failed to record observations: %s", exc)
    else:
        log.info("  Using synthetic climatology (no real station data available)")

    # ------------------------------------------------------------------
    # 5. Load recorded snapshots from previous sessions
    # ------------------------------------------------------------------
    log.info("Loading past snapshots from disk …")
    past_snaps = recorder.load_recent_snapshots(hours=config.TIMELINE_HISTORY_HOURS)
    if past_snaps:
        log.info("  Loaded %d past snapshots (up to %.0f h of history)",
                 len(past_snaps), config.TIMELINE_HISTORY_HOURS)
    else:
        log.info("  No past snapshots on disk — first session")

    # ------------------------------------------------------------------
    # 6. Build pre-computed timeline (present + future fast-forward)
    # ------------------------------------------------------------------
    log.info("Building timeline (%.0f h forecast) …", config.PREDICTION_HOURS)
    timeline = swarm.build_timeline(
        recorded_snapshots      = past_snaps,
        future_hours            = config.PREDICTION_HOURS,
        snapshot_interval_hours = config.TIMELINE_SNAPSHOT_INTERVAL_HOURS,
    )
    log.info("  Timeline: %d snapshots", len(timeline))

    # ------------------------------------------------------------------
    # 7. Background NOAA refresh loop
    # ------------------------------------------------------------------
    if not args.no_api:
        def _refresh_worker():
            while True:
                time.sleep(NOAA_REFRESH_INTERVAL)
                log.info("[NOAA refresh] Re-fetching station data …")
                obs = _fetch_noaa(config.SEED_STATIONS)
                if obs:
                    swarm.apply_observations(obs)
                    log.info("[NOAA refresh] Applied %d updated observations", len(obs))
                    try:
                        recorder.record_observations(
                            obs,
                            timestamp=datetime.datetime.now(datetime.timezone.utc),
                        )
                    except Exception:
                        pass

        t = threading.Thread(target=_refresh_worker, daemon=True, name="noaa-refresh")
        t.start()

    # ------------------------------------------------------------------
    # 8. Launch renderer
    # ------------------------------------------------------------------
    log.info("Opening visualization window …")
    from visualization.renderer import WeatherRenderer

    renderer = WeatherRenderer(
        swarm      = swarm,
        timeline   = timeline,
        mode       = "realtime",
        start_time = start_time,
        recorder   = recorder,
    )
    renderer.run()


if __name__ == "__main__":
    main()
