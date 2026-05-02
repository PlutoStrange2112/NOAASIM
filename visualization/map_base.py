"""
USA base map renderer.

Downloads the US states GeoJSON once, caches it locally, then draws
state boundaries as matplotlib PathPatch objects on a plain lat/lon axis.

No cartopy required — uses a simple equirectangular (plate carrée) projection
which is accurate enough for the continental US at this scale.
"""

import json
import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.path as mpath
import requests

log = logging.getLogger(__name__)

# Public-domain US state boundaries — Natural Earth / PublicaMundi
_GEOJSON_URL  = (
    "https://raw.githubusercontent.com/PublicaMundi/MappingAPI"
    "/master/data/geojson/us-states.json"
)
_CACHE_FILE   = os.path.join("cache", "us_states.geojson")

# Map aesthetics
_STATE_LINE_COLOR  = "#555566"
_STATE_FILL_COLOR  = "#1a1a2e"
_OCEAN_COLOR       = "#0d0d1a"
_COAST_LINE_COLOR  = "#7788aa"
_GRID_COLOR        = "#2a2a3e"


def _ensure_geojson():
    """Download and cache the GeoJSON file if not present."""
    os.makedirs("cache", exist_ok=True)
    if os.path.exists(_CACHE_FILE):
        return True
    log.info("Downloading US states GeoJSON …")
    try:
        r = requests.get(_GEOJSON_URL, timeout=15)
        r.raise_for_status()
        with open(_CACHE_FILE, "w", encoding="utf-8") as f:
            f.write(r.text)
        return True
    except Exception as exc:
        log.warning("Could not download GeoJSON: %s — map will be outline-only", exc)
        return False


def _geojson_to_patches(geojson_path):
    """
    Parse a GeoJSON FeatureCollection of Polygon / MultiPolygon features.
    Returns (fill_patches, border_patches) — fills at low zorder, borders
    drawn on top of the weather layer so state lines are always visible.
    """
    with open(geojson_path, encoding="utf-8") as f:
        data = json.load(f)

    fills   = []
    borders = []
    for feature in data.get("features", []):
        geo   = feature.get("geometry", {})
        gtype = geo.get("type", "")
        polys = []
        if gtype == "Polygon":
            polys = [geo["coordinates"]]
        elif gtype == "MultiPolygon":
            polys = geo["coordinates"]
        for poly in polys:
            for ring in poly:
                if len(ring) < 3:
                    continue
                verts = np.array(ring)  # (N, 2) lon, lat
                codes = ([mpath.Path.MOVETO]
                         + [mpath.Path.LINETO] * (len(verts) - 2)
                         + [mpath.Path.CLOSEPOLY])
                p = mpath.Path(verts, codes)
                fills.append(mpatches.PathPatch(
                    p, facecolor=_STATE_FILL_COLOR, edgecolor="none",
                    linewidth=0, zorder=1,
                ))
                borders.append(mpatches.PathPatch(
                    p, facecolor="none", edgecolor=_STATE_LINE_COLOR,
                    linewidth=0.7, zorder=7, alpha=0.85,
                ))
    return fills, borders


class USAMapBase:
    """
    Sets up a matplotlib figure with the continental USA drawn.

    After construction `self.fig` and `self.ax` are ready for overlay plots.
    The axis uses raw lon / lat coordinates.
    """

    def __init__(self, figsize=(16, 9), dpi=100):
        self.fig, self.ax = plt.subplots(figsize=figsize, dpi=dpi)
        self.fig.patch.set_facecolor(_OCEAN_COLOR)
        self.ax.set_facecolor(_OCEAN_COLOR)

        self._draw_base_map()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _draw_base_map(self):
        ax = self.ax
        from config import LAT_MIN, LAT_MAX, LON_MIN, LON_MAX

        ax.set_xlim(LON_MIN - 1, LON_MAX + 1)
        ax.set_ylim(LAT_MIN - 1, LAT_MAX + 1)
        ax.set_aspect("equal", adjustable="box")
        ax.set_facecolor(_OCEAN_COLOR)

        # latitude / longitude grid lines
        for lat in range(25, 52, 5):
            ax.axhline(lat, color=_GRID_COLOR, linewidth=0.4, zorder=0)
        for lon in range(-125, -65, 5):
            ax.axvline(lon, color=_GRID_COLOR, linewidth=0.4, zorder=0)

        # State patches
        if _ensure_geojson():
            try:
                fills, borders = _geojson_to_patches(_CACHE_FILE)
                for p in fills:
                    ax.add_patch(p)
                # Store borders; they must be re-added after weather layers
                self._border_patches = borders
                log.info("Drew %d state polygon patches", len(fills))
            except Exception as exc:
                log.warning("Could not draw state patches: %s", exc)
                self._draw_simple_outline()
        else:
            self._draw_simple_outline()

        # Tick labels
        ax.set_xticks(range(-120, -65, 10))
        ax.set_yticks(range(25, 52, 5))
        ax.set_xticklabels([f"{abs(x)}°W" for x in range(-120, -65, 10)],
                            fontsize=7, color="#8899aa")
        ax.set_yticklabels([f"{y}°N" for y in range(25, 52, 5)],
                            fontsize=7, color="#8899aa")
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_edgecolor("#334455")

    def _draw_simple_outline(self):
        """Fallback: draw a simple bounding rectangle."""
        from config import LAT_MIN, LAT_MAX, LON_MIN, LON_MAX
        rect = mpatches.Rectangle(
            (LON_MIN, LAT_MIN), LON_MAX - LON_MIN, LAT_MAX - LAT_MIN,
            facecolor=_STATE_FILL_COLOR, edgecolor=_COAST_LINE_COLOR,
            linewidth=1.0, zorder=1,
        )
        self.ax.add_patch(rect)

    # ------------------------------------------------------------------
    # Coordinate utilities
    # ------------------------------------------------------------------

    @staticmethod
    def latlon_to_xy(lat, lon):
        """Identity: we use raw lat/lon as plot coordinates."""
        return lon, lat
