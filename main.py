"""
NOAA AI Weather Visualization
==============================
Run:  python main.py           (window opens with timeline, NOAA loads in background)
      python main.py --no-api  (fully offline / synthetic only)
      python main.py --boids 500
      python main.py --full-history  (730-day FFT history — richer prediction)

Startup sequence:
  1. Synthetic seed → boids ready immediately
  2. FFT analysers built from 30-day history
  3. Past snapshots loaded from disk (previous sessions)
  4. Timeline fast-forwarded (120 hours into future) — this takes a few seconds
  5. Window opens: slider covers full history → present → future
  6. Background thread fetches NOAA live data and injects into swarm

Data grows every session:
  Each run records NOAA observations and live snapshots to cache/,
  which future sessions load for the historical half of the timeline.
"""

import argparse
import logging
import threading
import datetime
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)-8s %(name)s — %(message)s",
)
log = logging.getLogger("main")

parser = argparse.ArgumentParser(description="NOAA AI Weather Visualization")
parser.add_argument("--no-api",       action="store_true", help="Skip NOAA API calls")
parser.add_argument("--boids",        type=int, default=None, help="Override boid count")
parser.add_argument("--full-history", action="store_true",
                    help="Use 730-day FFT history (slower startup, richer prediction)")
args = parser.parse_args()


def main():
    import config
    if args.boids:
        config.NUM_BOIDS = args.boids

    rng        = np.random.default_rng(0)
    start_time = datetime.datetime.now(datetime.timezone.utc)

    # ------------------------------------------------------------------
    # 1. Build swarm with synthetic seed
    # ------------------------------------------------------------------
    log.info("Seeding %d weather boids from synthetic climatology …", config.NUM_BOIDS)
    from simulation.boids import WeatherSwarm
    swarm = WeatherSwarm(rng=rng)
    swarm.seed_synthetic(n=config.NUM_BOIDS)
    log.info("  Swarm ready: %d boids", len(swarm.boids))

    # ------------------------------------------------------------------
    # 2. Build FFT analysers
    # ------------------------------------------------------------------
    history_days = 730 if args.full_history else 30
    log.info("Building FFT analysers (%d-day history per boid) …", history_days)
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
    # 3. Load recorded snapshots from previous sessions
    # ------------------------------------------------------------------
    from data import recorder
    log.info("Loading past snapshots from disk …")
    past_snaps = recorder.load_recent_snapshots(hours=config.TIMELINE_HISTORY_HOURS)
    if past_snaps:
        log.info("  Loaded %d past snapshots (%.0f h of history)",
                 len(past_snaps), config.TIMELINE_HISTORY_HOURS)
    else:
        log.info("  No past snapshots on disk — first session")

    # ------------------------------------------------------------------
    # 4. Build pre-computed timeline (present + future fast-forward)
    # ------------------------------------------------------------------
    log.info("Building timeline (%.0f h future fast-forward) …",
             config.PREDICTION_HOURS)
    timeline = swarm.build_timeline(
        recorded_snapshots       = past_snaps,
        future_hours             = config.PREDICTION_HOURS,
        snapshot_interval_hours  = config.TIMELINE_SNAPSHOT_INTERVAL_HOURS,
    )
    log.info("  Timeline ready: %d snapshots", len(timeline))

    # ------------------------------------------------------------------
    # 5. Background NOAA fetch — injects real data while window is open
    #    and records observations to disk for future sessions
    # ------------------------------------------------------------------
    if not args.no_api:
        def _noaa_worker():
            log.info("[NOAA] Starting background station fetch …")
            try:
                from data.noaa_client import fetch_station_observations_parallel
                from config import SEED_STATIONS
                obs = fetch_station_observations_parallel(SEED_STATIONS)
                if obs:
                    swarm.apply_observations(obs)
                    log.info("[NOAA] Applied %d live observations to swarm", len(obs))
                    try:
                        recorder.record_observations(obs, timestamp=start_time)
                        log.info("[NOAA] Recorded %d observations to disk", len(obs))
                    except Exception as exc:
                        log.warning("[NOAA] Failed to record observations: %s", exc)
                else:
                    log.info("[NOAA] No observations returned — staying on synthetic data")
            except Exception as exc:
                log.warning("[NOAA] Background fetch failed: %s", exc)

        t = threading.Thread(target=_noaa_worker, daemon=True, name="noaa-fetch")
        t.start()
    else:
        log.info("--no-api set; running fully offline")

    # ------------------------------------------------------------------
    # 6. Launch renderer (blocks until window is closed)
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
