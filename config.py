# Continental USA bounds
LAT_MIN, LAT_MAX = 24.0, 50.0
LON_MIN, LON_MAX = -125.0, -66.0

# --- Boid simulation ---
NUM_BOIDS = 350
BOID_SEPARATION_RADIUS = 1.8   # degrees
BOID_ALIGNMENT_RADIUS  = 5.0   # degrees — fronts propagate this far
BOID_COHESION_RADIUS   = 9.0   # degrees — storm systems stay together

SEPARATION_WEIGHT        = 1.8
ALIGNMENT_WEIGHT         = 1.2
COHESION_WEIGHT          = 0.7
PRESSURE_GRADIENT_WEIGHT = 0.6
CORIOLIS_WEIGHT          = 0.25
FFT_ATTRACTOR_WEIGHT     = 0.15
TERRAIN_WEIGHT           = 0.4
GEOSTROPHIC_WEIGHT       = 0.35   # nudge toward geostrophic wind balance
LATENT_HEAT_WEIGHT       = 0.25   # warming/pressure-drop from precipitation

MAX_SPEED  = 1.8   # degrees/hour  (~200 km/hr)
MIN_SPEED  = 0.02
DAMPING    = 0.97

TRAIL_LENGTH = 25  # frames of position history per boid

# --- FFT analysis ---
DOMINANT_MODES        = 8     # frequency components to keep
PREDICTION_HOURS      = 120   # how far forward to predict
SYNTHETIC_HISTORY_DAYS = 730  # two years of synthetic history

# --- Terrain elevation grid (sampled at runtime, rough orography) ---
# Mountain ridges block / channel flow — encoded as simple Gaussian bumps
TERRAIN_FEATURES = [
    # (center_lat, center_lon, height_hPa, sigma_deg)
    (47.0, -121.5, 18.0, 2.5),   # Cascades
    (36.5, -118.5, 20.0, 3.0),   # Sierra Nevada
    (39.5, -106.0, 22.0, 3.5),   # Rockies (CO)
    (47.0, -113.5, 16.0, 3.0),   # Rockies (MT)
    (35.5, -83.5,   8.0, 2.0),   # Appalachians
    (46.5, -71.0,   6.0, 1.5),   # Adirondacks / Green Mtns
]

# --- NOAA API ---
NOAA_WEATHER_API  = "https://api.weather.gov"
NOAA_CDO_API      = "https://www.ncei.noaa.gov/cdo-web/api/v2"
NOAA_CDO_TOKEN    = ""   # optional — set for historical CDO data

# Key NOAA observation stations (ICAO / ASOS ids)
# Spread across the continental US for current-conditions seeding
SEED_STATIONS = [
    "KSEA", "KPDX", "KSFO", "KLAX", "KLAS", "KPHX", "KDEN",
    "KABQ", "KDFW", "KIAH", "KMSP", "KORD", "KSTL", "KMEM",
    "KATL", "KMIA", "KCLT", "KDCA", "KBOS", "KJFK", "KPHL",
    "KCLE", "KDET", "KPIT", "KBUF", "KOMA", "KMCI", "KSLC",
    "KBIL", "KBIS", "KSUX", "KTUL", "KLIT", "KJAN", "KMOB",
    # West Virginia — improves accuracy for the Monongalia/Marion county area
    "KCKB",   # North Central WV Airport (Clarksburg/Bridgeport — nearest to Fairmont)
    "KMGW",   # Morgantown Municipal (WVU area)
    "KCRW",   # Yeager Airport (Charleston)
    "KHTS",   # Tri-State Airport (Huntington)
    "KBKW",   # Raleigh County Memorial (Beckley)
]

# --- Timeline / playback ---
TIMELINE_SNAPSHOT_INTERVAL_HOURS = 1.0   # spacing between pre-built snapshots
TIMELINE_HISTORY_HOURS           = 72.0  # how far back to load from disk
PLAYBACK_SPEED_HOURS_PER_SEC     = 6.0   # simulated hours played per wall-clock second

# --- Visualization ---
FPS              = 60
SIM_DT_HOURS     = 0.5    # simulation hours per animation frame
BACKGROUND_ALPHA = 0.75
BOID_ARROW_SCALE = 0.4    # quiver scale factor
CONTOUR_LEVELS   = 12

# Color maps
CMAP_TEMP   = "RdBu_r"
CMAP_PRECIP = "Blues"
PRESSURE_COLOR = "#cccccc"

# Window
FIGURE_SIZE = (16, 9)
DPI         = 100
