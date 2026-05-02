"""
NOAA AI Weather Visualization
==============================
Run:  python main.py           (window opens immediately, NOAA loads in background)
      python main.py --no-api  (fully offline)
      python main.py --boids 500
      python main.py --full-history  (730-day FFT history instead of 30-day default)

Startup sequence (window opens in ~3-5 seconds):
  1. Synthetic seed → boids ready immediately
  2. FFT analysers built from 30-day history (fast default)
  3. Window opens with animated boids
  4. Background thread fetches NOAA live data and injects into swarm
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

    rng = np.random.default_rng(0)

    # ------------------------------------------------------------------
    # 1. Build swarm with synthetic seed (fast — window opens quickly)
    # ------------------------------------------------------------------
    log.info("Seeding %d weather boids from synthetic climatology …", config.NUM_BOIDS)
    from simulation.boids import WeatherSwarm
    swarm = WeatherSwarm(rng=rng)
    swarm.seed_synthetic(n=config.NUM_BOIDS)
    log.info("  Swarm ready: %d boids", len(swarm.boids))

    # ------------------------------------------------------------------
    # 2. Build FFT analysers (30-day default → ~2 s for 350 boids)
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
    # 3. Background NOAA fetch — injects real data while window is open
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
                    log.info("[NOAA] Applied %d live station observations to swarm", len(obs))
                else:
                    log.info("[NOAA] No observations returned — staying on synthetic data")
            except Exception as exc:
                log.warning("[NOAA] Background fetch failed: %s", exc)

        t = threading.Thread(target=_noaa_worker, daemon=True, name="noaa-fetch")
        t.start()
    else:
        log.info("--no-api set; running fully offline")

    # ------------------------------------------------------------------
    # 4. Launch renderer (blocks until window is closed)
    # ------------------------------------------------------------------
    log.info("Opening visualization window …")
    from visualization.renderer import WeatherRenderer

    renderer = WeatherRenderer(
        swarm      = swarm,
        mode       = "realtime",
        start_time = datetime.datetime.now(datetime.timezone.utc),
    )
    renderer.run()


if __name__ == "__main__":
    main()
