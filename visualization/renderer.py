"""
Weather visualization renderer and animation engine.

Layers (bottom to top):
  0  Ocean / map background  (map_base.py)
  1  State borders            (map_base.py)
  2  Temperature field        — smooth interpolated heatmap
  3  Pressure contours        — isobar lines
  4  Precipitation overlay    — blue wash where precip > threshold
  5  Boid trails              — faint polylines showing recent path
  6  Boid arrows              — quiver: direction = velocity, color = weather type
  7  Boid scatter             — dot at boid centre, colored by type
  8  Info panel               — title, time, legend

Uses matplotlib.animation.FuncAnimation for smooth playback.
A time slider lets the user scrub through past (historical), present,
and future (FFT-predicted) time.
"""

import datetime
import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.collections import LineCollection
from matplotlib.widgets import Slider, Button
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter

from visualization.map_base import USAMapBase
from simulation.boids import TYPE_COLORS, LOW, HIGH, FRONT_COLD, FRONT_WARM, CLEAR
from config import (
    LAT_MIN, LAT_MAX, LON_MIN, LON_MAX,
    CMAP_TEMP, CONTOUR_LEVELS, BACKGROUND_ALPHA,
    BOID_ARROW_SCALE, FIGURE_SIZE, DPI, SIM_DT_HOURS,
    TRAIL_LENGTH, FPS,
)

log = logging.getLogger(__name__)

# Interpolation grid resolution
_IGRID_LAT = 50
_IGRID_LON = 80

# Type → matplotlib color
_TYPE_CMAP = {
    LOW:        "#aa44ff",
    HIGH:       "#ff9933",
    FRONT_COLD: "#4488ff",
    FRONT_WARM: "#ff4444",
    CLEAR:      "#88ccee",
}
_TYPE_LABEL = {
    LOW: "Low Pressure", HIGH: "High Pressure",
    FRONT_COLD: "Cold Front", FRONT_WARM: "Warm Front", CLEAR: "Clear",
}


def _boid_colors(btypes):
    return [_TYPE_CMAP.get(t, "#ffffff") for t in btypes]


class WeatherRenderer:
    """
    Full-screen animated weather visualization.

    Parameters
    ----------
    swarm      : simulation.boids.WeatherSwarm
    mode       : 'realtime' | 'forecast'   (affects title)
    start_time : datetime for t=0 label
    """

    def __init__(self, swarm, mode="realtime", start_time=None):
        self.swarm      = swarm
        self.mode       = mode
        self.start_time = start_time or datetime.datetime.now(datetime.timezone.utc)
        self.paused       = False
        self._frame       = 0
        self._contour_tick = 0   # throttle: redraw contours every N frames

        # Interpolation grid
        self._grid_lon, self._grid_lat = np.meshgrid(
            np.linspace(LON_MIN, LON_MAX, _IGRID_LON),
            np.linspace(LAT_MIN, LAT_MAX, _IGRID_LAT),
        )

        # Build figure
        self._build_figure()

    # ------------------------------------------------------------------
    # Figure construction
    # ------------------------------------------------------------------

    def _build_figure(self):
        self._map = USAMapBase(figsize=FIGURE_SIZE, dpi=DPI)
        self.fig  = self._map.fig
        self.ax   = self._map.ax

        self.fig.subplots_adjust(left=0.04, right=0.82, bottom=0.12, top=0.95)

        # ---- Temperature background (imshow, updated each frame) ----
        dummy = np.zeros((_IGRID_LAT, _IGRID_LON))
        self._temp_img = self.ax.imshow(
            dummy,
            extent=[LON_MIN, LON_MAX, LAT_MIN, LAT_MAX],
            origin="lower", aspect="auto",
            cmap=CMAP_TEMP, alpha=BACKGROUND_ALPHA,
            vmin=-20, vmax=40, zorder=2,
            interpolation="bilinear",
        )

        # ---- Pressure contours (cleared and redrawn each frame) ----
        self._contour_artists = []   # QuadContourSet objects
        self._clabel_texts    = []   # clabel Text artists (removed separately)

        # ---- Precipitation overlay ----
        self._precip_img = self.ax.imshow(
            dummy,
            extent=[LON_MIN, LON_MAX, LAT_MIN, LAT_MAX],
            origin="lower", aspect="auto",
            cmap="Blues", alpha=0.0,   # alpha set per-frame from precip values
            vmin=0, vmax=5, zorder=3,
            interpolation="bilinear",
        )

        # ---- Trails — single LineCollection (replaces thousands of ax.plot calls) ----
        self._trail_coll = LineCollection(
            [], linewidths=0.7, alpha=0.35, zorder=4,
        )
        self.ax.add_collection(self._trail_coll)

        # ---- Boid arrows — each boid is an arrow pointing in wind direction ----
        snap = self.swarm.snapshot()
        spds = np.hypot(snap["vel_u"], snap["vel_v"]).clip(1e-6)
        arrow_scale = 2.2   # display length in data-unit degrees
        self._quiver = self.ax.quiver(
            snap["lons"], snap["lats"],
            snap["vel_u"] / spds * arrow_scale,
            snap["vel_v"] / spds * arrow_scale,
            color=_boid_colors(snap["btypes"]),
            scale=6.0, scale_units="xy", angles="xy",
            width=0.0025, headwidth=4, headlength=5,
            alpha=0.92, zorder=6,
        )

        # ---- State border overlay (drawn on top of weather layers) ----
        for p in getattr(self._map, "_border_patches", []):
            self.ax.add_patch(p)

        # ---- Colourbar ----
        cbar = self.fig.colorbar(
            self._temp_img,
            ax=self.ax, fraction=0.025, pad=0.01,
            label="Temperature (°C)",
        )
        cbar.ax.yaxis.label.set_color("#ccddee")
        cbar.ax.tick_params(colors="#ccddee")

        # ---- Legend (weather types) ----
        handles = [
            plt.Line2D([0], [0], marker="o", color="w",
                       markerfacecolor=c, markersize=8, label=l)
            for t, c in _TYPE_CMAP.items()
            for l in [_TYPE_LABEL[t]]
        ]
        self.ax.legend(
            handles=handles, loc="lower left",
            fontsize=7, framealpha=0.3,
            labelcolor="#ddeeff", facecolor="#111122",
            edgecolor="#334455",
        )

        # ---- Title ----
        self._title = self.fig.suptitle(
            self._make_title(),
            color="#ddeeff", fontsize=11, y=0.98,
        )

        # ---- Time slider ----
        ax_slider = self.fig.add_axes([0.05, 0.04, 0.65, 0.025])
        self._slider = Slider(
            ax_slider, "Time offset (hrs)", -48, 72,
            valinit=0, color="#334488", valstep=0.5,
        )
        self._slider.label.set_color("#aabbcc")
        self._slider.valtext.set_color("#aabbcc")

        # ---- Pause / play button ----
        ax_btn = self.fig.add_axes([0.73, 0.035, 0.08, 0.03])
        self._btn = Button(ax_btn, "[Pause]", color="#223344", hovercolor="#334455")
        self._btn.label.set_color("#ccddee")
        self._btn.on_clicked(self._toggle_pause)

        # ---- Info panel (right side) ----
        ax_info = self.fig.add_axes([0.83, 0.15, 0.16, 0.70])
        ax_info.set_facecolor("#0d0d1a")
        ax_info.set_xticks([]); ax_info.set_yticks([])
        for spine in ax_info.spines.values():
            spine.set_edgecolor("#334455")
        self._info_ax   = ax_info
        self._info_text = ax_info.text(
            0.05, 0.95, self._make_info(None),
            transform=ax_info.transAxes,
            va="top", ha="left",
            fontsize=7.5, color="#aabbcc",
            family="monospace",
        )

    # ------------------------------------------------------------------
    # Frame update
    # ------------------------------------------------------------------

    def _update(self, frame):
        if self.paused:
            return []

        self._frame = frame
        self.swarm.step()
        snap = self.swarm.snapshot()

        pts = np.column_stack([snap["lons"], snap["lats"]])

        # ---- Temperature field — nearest + gaussian smooth (2× faster than linear) ----
        grid_t = griddata(pts, snap["temp_c"], (self._grid_lon, self._grid_lat),
                          method="nearest")
        grid_t = gaussian_filter(grid_t.astype(float), sigma=2.5)
        self._temp_img.set_data(grid_t)

        # ---- Pressure contours — throttled to every 3 frames ----
        self._contour_tick = (self._contour_tick + 1) % 3
        if self._contour_tick == 0:
            for cs in self._contour_artists:
                cs.remove()
            for txt in self._clabel_texts:
                try:
                    txt.remove()
                except Exception:
                    pass
            self._contour_artists.clear()
            self._clabel_texts.clear()
            grid_p = griddata(pts, snap["pressure"], (self._grid_lon, self._grid_lat),
                              method="nearest")
            grid_p = gaussian_filter(grid_p.astype(float), sigma=1.5)
            try:
                cs = self.ax.contour(
                    self._grid_lon, self._grid_lat, grid_p,
                    levels=CONTOUR_LEVELS,
                    colors="#556677", linewidths=0.5, alpha=0.6, zorder=4,
                )
                labels = self.ax.clabel(cs, fmt="%d", fontsize=6, colors="#778899")
                self._contour_artists.append(cs)
                self._clabel_texts.extend(labels)
            except Exception:
                pass

        # ---- Precipitation overlay ----
        grid_pr = griddata(pts, snap["precip"], (self._grid_lon, self._grid_lat),
                           method="nearest", fill_value=0)
        self._precip_img.set_data(np.clip(grid_pr, 0, 5))
        max_pr = float(grid_pr.max())
        self._precip_img.set_alpha(min(0.45, max_pr * 0.15) if max_pr > 0.1 else 0.0)

        # ---- Trails — single LineCollection update (replaces 8750 ax.plot calls) ----
        segments  = []
        seg_colors = []
        for trail, btype in zip(snap["trails"], snap["btypes"]):
            if len(trail) < 2:
                continue
            trail_a = np.array(trail)
            xy = trail_a[:, ::-1]   # (T, 2) → (lon, lat) for x,y axes
            color = _TYPE_CMAP.get(int(btype), "#aaaaaa")
            for j in range(len(xy) - 1):
                segments.append([xy[j], xy[j + 1]])
                seg_colors.append(color)
        if segments:
            self._trail_coll.set_segments(segments)
            self._trail_coll.set_color(seg_colors)
        else:
            self._trail_coll.set_segments([])

        # ---- Boid arrows — unit-direction arrows coloured by weather type ----
        spds = np.hypot(snap["vel_u"], snap["vel_v"]).clip(1e-6)
        arrow_scale = 2.2
        self._quiver.set_offsets(np.column_stack([snap["lons"], snap["lats"]]))
        self._quiver.set_UVC(
            snap["vel_u"] / spds * arrow_scale,
            snap["vel_v"] / spds * arrow_scale,
        )
        self._quiver.set_color(_boid_colors(snap["btypes"]))

        # ---- Title & info ----
        self._title.set_text(self._make_title())
        self._info_text.set_text(self._make_info(snap))

        return []

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_title(self):
        dt    = self.start_time + datetime.timedelta(hours=self.swarm.hour_elapsed)
        phase = "FORECAST" if self.swarm.hour_elapsed > 0 else "CURRENT"
        return (
            f"NOAA AI Weather System  ·  {phase}  ·  "
            f"{dt.strftime('%Y-%m-%d  %H:%M UTC')}  "
            f"(T+{self.swarm.hour_elapsed:.1f} h)"
        )

    def _make_info(self, snap):
        lines = ["── WEATHER TYPES ──\n"]
        if snap is not None:
            btypes = snap["btypes"]
            for t, label in _TYPE_LABEL.items():
                cnt = int((btypes == t).sum())
                if cnt > 0:
                    lines.append(f"{label[:12]:<12} {cnt:3d}")
            lines.append("")
            lines.append("── FIELD STATS ──\n")
            lines.append(f"Temp  {snap['temp_c'].min():.0f}..{snap['temp_c'].max():.0f} °C")
            lines.append(f"Pres  {snap['pressure'].min():.0f}..{snap['pressure'].max():.0f} hPa")
            lines.append(f"Humid {snap['humidity'].mean():.0f}% avg")
            raining = int((snap["precip"] > 0.1).sum())
            lines.append(f"Precip {raining} parcels")
            lines.append("")
            lines.append("── FFT MODES ──\n")
            lines.append(f"Dominant: {self.swarm.hour_elapsed:.1f} h")
            lines.append("in future")
            lines.append("")
            lines.append("── CONTROLS ──\n")
            lines.append("Slider: time offset")
            lines.append("Button: pause/play")
        return "\n".join(lines)

    def _toggle_pause(self, event):
        self.paused = not self.paused
        # Access event_source directly — FuncAnimation.pause()/resume() crash
        # when event_source is None (e.g. before the Tk mainloop initialises it).
        if hasattr(self, "_anim"):
            es = getattr(self._anim, "event_source", None)
            if es is not None:
                if self.paused:
                    es.stop()
                else:
                    es.start()
        self._btn.label.set_text("[ Play ]" if self.paused else "[Pause]")
        self.fig.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Launch
    # ------------------------------------------------------------------

    def run(self, n_frames=None):
        """Start the animation loop."""
        if n_frames is None:
            from config import PREDICTION_HOURS
            n_frames = int(PREDICTION_HOURS / SIM_DT_HOURS) + 200

        self._anim = animation.FuncAnimation(
            self.fig, self._update,
            frames=n_frames,
            interval=max(16, 1000 // FPS),
            blit=False,
            repeat=False,
        )
        plt.show()
