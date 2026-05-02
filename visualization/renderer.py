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

Timeline / scrubber
-------------------
  • `timeline` is a sorted list of {"hour_offset": float, "snap": dict}
    covering past recordings (hour_offset < 0) and fast-forward future
    (hour_offset > 0).  hour_offset = 0 is the moment the session started.
  • The time slider maps to hour_offset.  Dragging it scrubs to the nearest
    available snapshot.
  • In play mode the playhead advances at PLAYBACK_SPEED_HOURS_PER_SEC using
    wall-clock time so the speed is independent of FPS.
  • While playing, the slider tracks the playhead automatically.
  • Snapshots recorded during the session are appended to the timeline so the
    live view "extends" the tape forward.
"""

import time
import datetime
import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors
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
    PLAYBACK_SPEED_HOURS_PER_SEC,
)

log = logging.getLogger(__name__)

_IGRID_LAT = 50
_IGRID_LON = 80

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
    Full-screen animated weather visualization with a video-scrubber slider.

    Parameters
    ----------
    swarm      : simulation.boids.WeatherSwarm
    timeline   : list of {"hour_offset": float, "snap": dict}, sorted by
                 hour_offset.  Built by WeatherSwarm.build_timeline().
    mode       : 'realtime' | 'forecast'
    start_time : datetime for t=0 label
    recorder   : optional data.recorder module for live snapshot recording
    """

    def __init__(self, swarm, timeline=None, mode="realtime",
                 start_time=None, recorder=None):
        self.swarm      = swarm
        self.mode       = mode
        self.start_time = start_time or datetime.datetime.now(datetime.timezone.utc)
        self.recorder   = recorder

        self.paused     = False
        self._frame     = 0
        self._contour_tick = 0

        # ---- Timeline state ----
        self._timeline  = timeline if timeline else []
        self._playhead  = self._find_present_index()   # index into _timeline
        self._last_advance_wall = time.monotonic()      # wall clock at last step

        # Track whether slider is being dragged (suppress playhead auto-update)
        self._slider_dragging = False
        self._record_frame_interval = 60   # record snapshot every N frames

        # Interpolation grid
        self._grid_lon, self._grid_lat = np.meshgrid(
            np.linspace(LON_MIN, LON_MAX, _IGRID_LON),
            np.linspace(LAT_MIN, LAT_MAX, _IGRID_LAT),
        )

        self._build_figure()

    # ------------------------------------------------------------------
    # Timeline helpers
    # ------------------------------------------------------------------

    def _find_present_index(self):
        """Return the index whose hour_offset is closest to 0."""
        if not self._timeline:
            return 0
        return min(range(len(self._timeline)),
                   key=lambda i: abs(self._timeline[i]["hour_offset"]))

    def _hour_offset_at(self, idx):
        if not self._timeline:
            return 0.0
        idx = max(0, min(idx, len(self._timeline) - 1))
        return self._timeline[idx]["hour_offset"]

    def _snap_at(self, idx):
        if not self._timeline:
            return None
        idx = max(0, min(idx, len(self._timeline) - 1))
        return self._timeline[idx]["snap"]

    def _nearest_index(self, hour_offset):
        """Binary-search for the timeline index closest to hour_offset."""
        if not self._timeline:
            return 0
        lo, hi = 0, len(self._timeline) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if self._timeline[mid]["hour_offset"] < hour_offset:
                lo = mid + 1
            else:
                hi = mid
        # compare lo with lo-1
        if lo > 0:
            if (abs(self._timeline[lo - 1]["hour_offset"] - hour_offset) <
                    abs(self._timeline[lo]["hour_offset"] - hour_offset)):
                return lo - 1
        return lo

    def _append_live_snap(self, snap, hour_offset):
        """Append a live snapshot to the end of the timeline."""
        entry = {"hour_offset": hour_offset, "snap": snap}
        self._timeline.append(entry)

    # ------------------------------------------------------------------
    # Figure construction
    # ------------------------------------------------------------------

    def _build_figure(self):
        self._map = USAMapBase(figsize=FIGURE_SIZE, dpi=DPI)
        self.fig  = self._map.fig
        self.ax   = self._map.ax

        self.fig.subplots_adjust(left=0.04, right=0.82, bottom=0.12, top=0.95)

        dummy = np.zeros((_IGRID_LAT, _IGRID_LON))

        # Temperature background
        self._temp_img = self.ax.imshow(
            dummy,
            extent=[LON_MIN, LON_MAX, LAT_MIN, LAT_MAX],
            origin="lower", aspect="auto",
            cmap=CMAP_TEMP, alpha=BACKGROUND_ALPHA,
            vmin=-20, vmax=40, zorder=2,
            interpolation="bilinear",
        )

        # Pressure contours
        self._contour_artists = []
        self._clabel_texts    = []

        # Precipitation overlay
        self._precip_img = self.ax.imshow(
            dummy,
            extent=[LON_MIN, LON_MAX, LAT_MIN, LAT_MAX],
            origin="lower", aspect="auto",
            cmap="Blues", alpha=0.0,
            vmin=0, vmax=5, zorder=3,
            interpolation="bilinear",
        )

        # Trails
        self._trail_coll = LineCollection(
            [], linewidths=0.7, alpha=0.35, zorder=4,
        )
        self.ax.add_collection(self._trail_coll)

        # Boid arrows
        snap0 = self._snap_at(self._playhead) or self.swarm.snapshot()
        spds  = np.hypot(snap0["vel_u"], snap0["vel_v"]).clip(1e-6)
        arrow_scale = 2.2
        self._quiver = self.ax.quiver(
            snap0["lons"], snap0["lats"],
            snap0["vel_u"] / spds * arrow_scale,
            snap0["vel_v"] / spds * arrow_scale,
            color=_boid_colors(snap0["btypes"]),
            scale=6.0, scale_units="xy", angles="xy",
            width=0.0025, headwidth=4, headlength=5,
            alpha=0.92, zorder=6,
        )

        # State borders (on top of weather layers)
        for p in getattr(self._map, "_border_patches", []):
            self.ax.add_patch(p)

        # Colourbar
        cbar = self.fig.colorbar(
            self._temp_img,
            ax=self.ax, fraction=0.025, pad=0.01,
            label="Temperature (°C)",
        )
        cbar.ax.yaxis.label.set_color("#ccddee")
        cbar.ax.tick_params(colors="#ccddee")

        # Legend
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

        # Title
        self._title = self.fig.suptitle(
            self._make_title(0.0),
            color="#ddeeff", fontsize=11, y=0.98,
        )

        # ---- Time slider ----
        first_h = self._timeline[0]["hour_offset"]  if self._timeline else -48.0
        last_h  = self._timeline[-1]["hour_offset"] if self._timeline else 72.0
        ax_slider = self.fig.add_axes([0.05, 0.04, 0.65, 0.025])
        self._slider = Slider(
            ax_slider, "Time offset (hrs)", first_h, last_h,
            valinit=self._hour_offset_at(self._playhead),
            color="#334488", valstep=None,
        )
        self._slider.label.set_color("#aabbcc")
        self._slider.valtext.set_color("#aabbcc")
        self._slider_cid = self._slider.on_changed(self._on_slider_changed)

        # ---- Pause / Play button ----
        ax_btn = self.fig.add_axes([0.73, 0.035, 0.08, 0.03])
        self._btn = Button(ax_btn, "[Pause]", color="#223344", hovercolor="#334455")
        self._btn.label.set_color("#ccddee")
        self._btn.on_clicked(self._toggle_pause)

        # ---- Timeline label (past / live / future indicator) ----
        ax_tlabel = self.fig.add_axes([0.73, 0.07, 0.08, 0.025])
        ax_tlabel.set_axis_off()
        self._tlabel = ax_tlabel.text(
            0.5, 0.5, self._time_zone_label(self._hour_offset_at(self._playhead)),
            ha="center", va="center", fontsize=7.5,
            color="#aabbcc", transform=ax_tlabel.transAxes,
        )

        # ---- Info panel ----
        ax_info = self.fig.add_axes([0.83, 0.15, 0.16, 0.70])
        ax_info.set_facecolor("#0d0d1a")
        ax_info.set_xticks([]); ax_info.set_yticks([])
        for spine in ax_info.spines.values():
            spine.set_edgecolor("#334455")
        self._info_ax   = ax_info
        self._info_text = ax_info.text(
            0.05, 0.95, self._make_info(None, 0.0),
            transform=ax_info.transAxes,
            va="top", ha="left",
            fontsize=7.5, color="#aabbcc",
            family="monospace",
        )

        # Render initial frame
        self._render_from_snap(snap0, self._hour_offset_at(self._playhead))

    # ------------------------------------------------------------------
    # Slider callback
    # ------------------------------------------------------------------

    def _on_slider_changed(self, val):
        """Called when user drags the slider."""
        self._slider_dragging = True
        idx  = self._nearest_index(val)
        snap = self._snap_at(idx)
        if snap is None:
            snap = self.swarm.snapshot()
        self._playhead = idx
        self._render_from_snap(snap, val)
        self.fig.canvas.draw_idle()
        self._slider_dragging = False

    # ------------------------------------------------------------------
    # Frame update (FuncAnimation callback)
    # ------------------------------------------------------------------

    def _update(self, frame):
        if self.paused:
            return []

        self._frame = frame

        now_wall = time.monotonic()
        elapsed  = now_wall - self._last_advance_wall

        # Advance playhead based on wall-clock time and playback speed
        hours_advance = elapsed * PLAYBACK_SPEED_HOURS_PER_SEC
        if self._timeline:
            cur_hour = self._hour_offset_at(self._playhead)
            target_hour = cur_hour + hours_advance
            new_idx = self._nearest_index(target_hour)
            if new_idx != self._playhead:
                self._playhead = new_idx
                self._last_advance_wall = now_wall

            # If we've reached the end of the pre-built timeline, extend with
            # live swarm steps so playback continues seamlessly.
            at_end = (self._playhead >= len(self._timeline) - 1)
        else:
            at_end = True

        if at_end:
            # Step the live swarm and append a new entry
            self.swarm.step()
            live_snap   = self.swarm.snapshot()
            live_hour   = self.swarm.hour_elapsed
            self._append_live_snap(live_snap, live_hour)
            self._playhead = len(self._timeline) - 1
            self._last_advance_wall = now_wall

            # Periodically record to disk
            if self.recorder and frame % self._record_frame_interval == 0:
                abs_time = self.start_time + datetime.timedelta(hours=live_hour)
                try:
                    self.recorder.record_snapshot(live_snap, abs_time)
                except Exception:
                    pass

        snap      = self._snap_at(self._playhead)
        hour      = self._hour_offset_at(self._playhead)

        self._render_from_snap(snap, hour)

        # Keep slider in sync (without triggering on_changed)
        if not self._slider_dragging:
            self._slider.eventson = False
            self._slider.set_val(float(hour))
            self._slider.eventson = True

        return []

    # ------------------------------------------------------------------
    # Core rendering — shared by update() and slider callback
    # ------------------------------------------------------------------

    def _render_from_snap(self, snap, hour_offset):
        if snap is None:
            return

        pts = np.column_stack([snap["lons"], snap["lats"]])

        # Temperature
        grid_t = griddata(pts, snap["temp_c"], (self._grid_lon, self._grid_lat),
                          method="nearest")
        grid_t = gaussian_filter(grid_t.astype(float), sigma=2.5)
        self._temp_img.set_data(grid_t)

        # Pressure contours (throttled to every 3 calls from animation)
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

        # Precipitation
        grid_pr = griddata(pts, snap["precip"], (self._grid_lon, self._grid_lat),
                           method="nearest", fill_value=0)
        self._precip_img.set_data(np.clip(grid_pr, 0, 5))
        max_pr = float(grid_pr.max())
        self._precip_img.set_alpha(min(0.45, max_pr * 0.15) if max_pr > 0.1 else 0.0)

        # Trails
        segments   = []
        seg_colors = []
        for trail, btype in zip(snap["trails"], snap["btypes"]):
            if len(trail) < 2:
                continue
            trail_a = np.array(trail)
            xy      = trail_a[:, ::-1]
            color   = _TYPE_CMAP.get(int(btype), "#aaaaaa")
            for j in range(len(xy) - 1):
                segments.append([xy[j], xy[j + 1]])
                seg_colors.append(color)
        if segments:
            self._trail_coll.set_segments(segments)
            self._trail_coll.set_color(seg_colors)
        else:
            self._trail_coll.set_segments([])

        # Boid arrows
        spds = np.hypot(snap["vel_u"], snap["vel_v"]).clip(1e-6)
        arrow_scale = 2.2
        self._quiver.set_offsets(np.column_stack([snap["lons"], snap["lats"]]))
        self._quiver.set_UVC(
            snap["vel_u"] / spds * arrow_scale,
            snap["vel_v"] / spds * arrow_scale,
        )
        self._quiver.set_color(_boid_colors(snap["btypes"]))

        # Title, zone label, info panel
        self._title.set_text(self._make_title(hour_offset))
        self._tlabel.set_text(self._time_zone_label(hour_offset))
        self._info_text.set_text(self._make_info(snap, hour_offset))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_title(self, hour_offset):
        dt    = self.start_time + datetime.timedelta(hours=hour_offset)
        if hour_offset < -0.25:
            phase = f"HISTORY  T{hour_offset:+.1f} h"
        elif hour_offset < 0.25:
            phase = "LIVE NOW"
        else:
            phase = f"FORECAST T+{hour_offset:.1f} h"
        return (
            f"NOAA AI Weather System  ·  {phase}  ·  "
            f"{dt.strftime('%Y-%m-%d  %H:%M UTC')}"
        )

    def _time_zone_label(self, hour_offset):
        if hour_offset < -0.25:
            return "[ HISTORY ]"
        if hour_offset < 0.25:
            return "[  LIVE   ]"
        return "[ FORECAST]"

    def _make_info(self, snap, hour_offset):
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
            lines.append("── TIMELINE ──\n")
            n   = len(self._timeline)
            idx = self._playhead
            lines.append(f"Snap {idx+1}/{n}")
            h0  = self._timeline[0]["hour_offset"]  if self._timeline else 0.0
            h1  = self._timeline[-1]["hour_offset"] if self._timeline else 0.0
            lines.append(f"Range {h0:+.0f}..{h1:+.0f} h")
            lines.append("")
            lines.append("── CONTROLS ──\n")
            lines.append("Slider: scrub time")
            lines.append("Button: pause/play")
        return "\n".join(lines)

    def _toggle_pause(self, event):
        self.paused = not self.paused
        if hasattr(self, "_anim"):
            es = getattr(self._anim, "event_source", None)
            if es is not None:
                if self.paused:
                    es.stop()
                else:
                    self._last_advance_wall = time.monotonic()
                    es.start()
        self._btn.label.set_text("[ Play ]" if self.paused else "[Pause]")
        self.fig.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Launch
    # ------------------------------------------------------------------

    def run(self, n_frames=None):
        """Start the animation loop."""
        if n_frames is None:
            # run "forever" — effectively unlimited frames
            n_frames = 1_000_000

        self._anim = animation.FuncAnimation(
            self.fig, self._update,
            frames=n_frames,
            interval=max(16, 1000 // FPS),
            blit=False,
            repeat=False,
        )
        plt.show()
