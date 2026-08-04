#!/usr/bin/env python3
# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Closed-loop routing demo on RoadWeaver-generated maps.

Evidence for "maps can be directly deployed in driving simulators, supporting
scalable closed-loop evaluation":

* Fig A — route-planning grid: tactics2d ``Router`` plans long routes spanning
  each generated map (orange route, green start, blue goal); the map is drawn
  with the official tactics2d ``MatplotlibRenderer``.
* Fig B — closed-loop driving: an ego vehicle (kinematic bicycle + pure-pursuit
  follower) drives a long planned route; trajectory overlay + lateral / speed /
  steering panels.
* Fig C — scalability summary: routing success rate and route length across
  generated maps.

Usage:
    conda activate road-weaver
    python scripts/visualize_router_closed_loop.py
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tactics2d"))

from network_generator.pipeline import build_map  # noqa: E402
from tactics2d.display.renderers import MatplotlibRenderer  # noqa: E402
from tactics2d.participant.trajectory.state import State  # noqa: E402
from tactics2d.physics.single_track_kinematics import SingleTrackKinematics  # noqa: E402
from tactics2d.routing.algorithm_adapter import AlgorithmAdapter  # noqa: E402
from tactics2d.routing.graph_builder import GraphBuilder  # noqa: E402
from tactics2d.routing.router import Router  # noqa: E402
from tactics2d.routing.utils import find_nearest_lane, get_lane_centerline  # noqa: E402
from utils.render import map_to_road_elements  # noqa: E402

OUT = REPO / "analysis" / "router_closed_loop"
OUT.mkdir(parents=True, exist_ok=True)

ROUTER = Router(algorithm="dijkstra", include_neighbors=True, cost_mode="distance")


class _Route:
    """Lightweight route result (path / lane_ids / total_cost)."""

    def __init__(self, lane_ids, path, cost):
        self.lane_ids = lane_ids
        self.path = path
        self.total_cost = cost
        self.is_empty = len(lane_ids) == 0


class CachedRouter:
    """Router over a prebuilt routing graph, reused across many plans.

    ``Router.plan`` rebuilds the lane graph on every call; for batch planning
    (long-route sampling, scalability sweep) that is wasteful, so this class
    builds the graph once and runs Dijkstra against it.
    """

    def __init__(self, hd):
        self.hd = hd
        self.rg = GraphBuilder(
            include_neighbors=True, lane_change_penalty=0.0, cost_mode="distance"
        ).build(hd)

    def plan(self, start, goal):
        sl = find_nearest_lane(self.hd, start)
        gl = find_nearest_lane(self.hd, goal)
        if sl is None or gl is None:
            return _Route([], None, 0.0)
        if sl not in self.rg.lane_id_to_index or gl not in self.rg.lane_id_to_index:
            return _Route([], None, 0.0)
        si, gi = self.rg.lane_id_to_index[sl], self.rg.lane_id_to_index[gl]
        path_idx, cost, _ = AlgorithmAdapter.dijkstra(self.rg, si, gi)
        if not path_idx:
            return _Route([], None, 0.0)
        lane_ids = [self.rg.index_to_lane_id[idx] for idx in path_idx]
        segs = [get_lane_centerline(self.hd.lanes[lid]) for lid in lane_ids]
        segs = [s for s in segs if s is not None]
        path = np.concatenate(segs) if segs else None
        return _Route(lane_ids, path, float(cost))


# ── Rendering with the official tactics2d renderer ────────────────────────


def render_map_t2d(ax, hd, sensor=None):
    """Draw a generated map with the official MatplotlibRenderer onto *ax*.

    The renderer draws roads / junctions as solid black shapes; the returned
    bounds let callers overlay routes with the same coordinate frame.
    """
    elems = map_to_road_elements(hd)
    xs, ys = [], []
    for el in elems:
        for p in el["geometry"]:
            xs.append(p[0])
            ys.append(p[1])
    if not xs:
        ax.set_aspect("equal")
        ax.set_axis_off()
        return None, None
    xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2

    renderer = MatplotlibRenderer(
        resolution=(100, 100), xlim=(xmin, xmax), ylim=(ymin, ymax), dpi=100, auto_scale=False
    )
    orig_fig = renderer.fig
    renderer.fig = ax.figure
    renderer.ax = ax
    plt.close(orig_fig)  # discard the throwaway figure

    geometry_data = {
        "metadata": {"sensor_position": sensor or [cx, cy], "sensor_yaw": 0.0},
        "map_data": {"road_id_to_remove": [], "road_elements": elems},
        "participant_data": {
            "participant_id_to_create": [],
            "participant_id_to_remove": [],
            "participants": [],
        },
    }
    renderer.update(geometry_data)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    return (xmin, xmax), (ymin, ymax)


# ── Long-route sampling ───────────────────────────────────────────────────


def sample_long_route(hd, rng, tries=100, min_len_m=900.0, max_len_m=2200.0):
    """Sample lane pairs and return the *longest* successful route.

    Bias toward far-apart roads: of all routable pairs tried, keep the one
    with the greatest route length (``total_cost`` in distance mode).
    """
    lanes = [l for l in hd.lanes if l.startswith("e")]
    if len(lanes) < 2:
        return None
    router = CachedRouter(hd)
    best = None
    for _ in range(tries):
        a, b = rng.choice(lanes, 2, replace=False)
        cla = get_lane_centerline(hd.lanes[a])
        clb = get_lane_centerline(hd.lanes[b])
        if cla is None or clb is None or len(cla) < 2 or len(clb) < 2:
            continue
        start = np.asarray(cla[0])
        goal = np.asarray(clb[len(clb) // 2])
        route = router.plan(start, goal)
        if route.path is None or len(route.path) < 3 or route.is_empty:
            continue
        plen = float(route.total_cost)
        if plen < min_len_m or plen > max_len_m:
            continue
        if best is None or plen > best[3]:
            best = (route, start, goal, plen)
    if best is None:
        return None
    return best[0], best[1], best[2]


# ── Fig A: route-planning grid ────────────────────────────────────────────


def fig_route_grid(seeds=(0, 1, 2, 3, 4, 5), out=None):
    fig, axes = plt.subplots(2, 3, figsize=(17, 12))
    rng = np.random.default_rng(7)
    stats = []
    for k, seed in enumerate(seeds):
        ax = axes[k // 3, k % 3]
        hd = build_map(seed=seed)
        res = sample_long_route(hd, rng, tries=50, min_len_m=1000.0, max_len_m=2200.0)
        bounds = render_map_t2d(ax, hd)
        if bounds is None or res is None:
            ax.set_title(f"seed {seed}: no long routable pair", fontsize=10)
            stats.append(None)
            continue
        route, start, goal = res
        path = np.asarray(route.path)
        ax.plot(path[:, 0], path[:, 1], color="#e67e22", lw=3.5, alpha=0.95, zorder=6)
        ax.plot(start[0], start[1], "o", color="#27ae60", ms=13, mec="black", mew=1.4, zorder=7)
        ax.plot(goal[0], goal[1], "o", color="#2980b9", ms=13, mec="black", mew=1.4, zorder=7)
        ax.set_title(
            f"seed {seed}  •  {len(route.lane_ids)} lanes  •  {route.total_cost:.0f} m", fontsize=10
        )
        stats.append((seed, len(route.lane_ids), route.total_cost))
    fig.suptitle(
        "RoadWeaver-generated maps  →  tactics2d Router long-range planning", fontsize=14, y=0.99
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = out or (OUT / "figA_route_grid.png")
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[demo] figA -> {path}")
    return stats


# ── Fig B: closed-loop driving ────────────────────────────────────────────


def _pure_pursuit_step(path, cum, pos, heading, lookahead):
    """Return steering angle to the lookahead point on ``path``."""
    d = np.linalg.norm(path - np.asarray(pos), axis=1)
    idx = int(d.argmin())
    s_target = cum[idx] + lookahead
    j = int(np.searchsorted(cum, s_target))
    j = min(j, len(path) - 1)
    target = path[j]
    alpha = np.arctan2(target[1] - pos[1], target[0] - pos[0]) - heading
    alpha = (alpha + np.pi) % (2 * np.pi) - np.pi  # normalize to [-pi, pi]
    wheelbase = 2.8
    steer = np.arctan2(2.0 * wheelbase * np.sin(alpha), max(lookahead, 1e-3))
    return float(np.clip(steer, -np.pi / 4, np.pi / 4))


def follow_route(hd, route, v_target=15.0, dt=0.1, max_t=None):
    """Drive a kinematic-bicycle ego along ``route.path``; return trajectory dict."""
    path = np.asarray(route.path)
    cum = np.cumsum(np.r_[0.0, np.linalg.norm(np.diff(path, axis=0), axis=1)])
    total = float(cum[-1])
    if max_t is None:
        max_t = max(60.0, 1.5 * total / v_target)  # adapt to route length

    physics = SingleTrackKinematics(
        lf=1.3,
        lr=1.5,
        steer_range=(-np.pi / 4, np.pi / 4),
        speed_range=(0.0, 40.0),
        accel_range=(-8.0, 5.0),
        interval=100,
    )
    h0 = float(np.arctan2(path[1, 1] - path[0, 1], path[1, 0] - path[0, 0]))
    state = State(frame=0, x=float(path[0, 0]), y=float(path[0, 1]), heading=h0, speed=v_target)

    xs, ys, heads, speeds, steers, times, s_along = [], [], [], [], [], [], []
    n_steps = int(max_t / dt)
    for k in range(n_steps):
        pos = np.asarray(state.location)
        h = float(state.heading)
        v = float(state.speed)
        lookahead = max(6.0, 0.8 * v)
        steer = _pure_pursuit_step(path, cum, pos, h, lookahead)
        accel = float(np.clip(1.2 * (v_target - v), -8.0, 5.0))
        state, _, _ = physics.step(state, accel, steer, interval=int(dt * 1000))

        xs.append(pos[0])
        ys.append(pos[1])
        heads.append(h)
        speeds.append(v)
        steers.append(steer)
        times.append(k * dt)

        d = np.linalg.norm(path - np.asarray(state.location), axis=1)
        s_along.append(float(cum[int(d.argmin())]))
        if s_along[-1] >= 0.98 * total:
            break

    traj = {
        "x": np.array(xs),
        "y": np.array(ys),
        "heading": np.array(heads),
        "speed": np.array(speeds),
        "steer": np.array(steers),
        "t": np.array(times),
        "s_along": np.array(s_along),
        "path": path,
        "cum": cum,
        "total": total,
    }
    return traj


def fig_closed_loop(seed=2, out=None):
    hd = build_map(seed=seed)
    rng = np.random.default_rng(11)
    res = sample_long_route(hd, rng, tries=50, min_len_m=900.0, max_len_m=1800.0)
    if res is None:
        print(f"[demo] figB: no long routable pair for seed {seed}")
        return None
    route, start, goal = res
    traj = follow_route(hd, route)

    fig = plt.figure(figsize=(16, 7.5))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.6, 1, 1], height_ratios=[1, 1])
    ax0 = fig.add_subplot(gs[:, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])
    ax3 = fig.add_subplot(gs[1, 1])
    ax4 = fig.add_subplot(gs[1, 2])

    bounds = render_map_t2d(ax0, hd)
    path = traj["path"]
    ax0.plot(
        path[:, 0],
        path[:, 1],
        color="#e67e22",
        lw=2.5,
        ls="--",
        alpha=0.95,
        zorder=6,
        label="planned route",
    )
    pts = np.stack([traj["x"], traj["y"]], axis=1)
    segs = np.stack([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(segs, cmap="autumn", linewidth=2.8, zorder=7)
    lc.set_array(traj["t"])
    ax0.add_collection(lc)
    ax0.plot(start[0], start[1], "o", color="#27ae60", ms=12, mec="black", mew=1.4, zorder=8)
    ax0.plot(goal[0], goal[1], "o", color="#2980b9", ms=12, mec="black", mew=1.4, zorder=8)
    ax0.legend(loc="upper right", fontsize=9)
    prog = traj["s_along"][-1] / traj["total"] * 100
    ax0.set_title(
        f"seed {seed}  •  route {traj['total']:.0f} m  •  driven {prog:.0f}%", fontsize=11
    )

    ax1.plot(traj["t"], traj["speed"], color="#16a085", lw=1.8)
    ax1.set_title("speed (m/s)")
    ax1.set_xlabel("t (s)")
    ax2.plot(traj["t"], traj["steer"] * 180 / np.pi, color="#c0392b", lw=1.8)
    ax2.set_title("steering (deg)")
    ax2.set_xlabel("t (s)")
    lat = np.array(
        [
            np.min(np.linalg.norm(traj["path"] - p, axis=1))
            for p in np.stack([traj["x"], traj["y"]], axis=1)
        ]
    )
    ax3.plot(traj["t"], lat, color="#8e44ad", lw=1.8)
    ax3.set_title("lateral deviation (m)")
    ax3.set_xlabel("t (s)")
    ax4.plot(traj["s_along"] / traj["total"] * 100, color="#2980b9", lw=1.8)
    ax4.set_title("progress along route (%)")
    ax4.set_xlabel("t (s)")
    for ax in (ax1, ax2, ax3, ax4):
        ax.grid(alpha=0.3)

    fig.suptitle(
        "Closed-loop: ego vehicle follows a long Router path on a generated map", fontsize=14, y=1.0
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path_out = out or (OUT / "figB_closed_loop.png")
    fig.savefig(path_out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[demo] figB -> {path_out}  (completion {prog:.0f}%, mean lat {lat.mean():.2f} m)")
    return {"completion": prog, "lat_mean": float(lat.mean()), "route_len": traj["total"]}


# ── GIF: ego driving animation ────────────────────────────────────────────


def make_trace_gif(seed=2, out=None, n_frames=24):
    """Render an ego-driving GIF (bonus closed-loop animation)."""
    hd = build_map(seed=seed)
    rng = np.random.default_rng(11)
    res = sample_long_route(hd, rng, tries=50, min_len_m=900.0, max_len_m=1800.0)
    if res is None:
        print(f"[demo] gif: no long routable pair for seed {seed}")
        return
    route, start, goal = res
    traj = follow_route(hd, route)

    path = traj["path"]
    xmin, xmax = path[:, 0].min() - 120, path[:, 0].max() + 120
    ymin, ymax = path[:, 1].min() - 120, path[:, 1].max() + 120

    frames = []
    n = len(traj["x"])
    idx = np.linspace(0, n - 1, n_frames).astype(int)
    for k in idx:
        fig, ax = plt.subplots(figsize=(7, 7))
        bounds = render_map_t2d(ax, hd)
        ax.plot(path[:, 0], path[:, 1], color="#e67e22", lw=2.2, ls="--", alpha=0.9, zorder=6)
        ax.plot(traj["x"][: k + 1], traj["y"][: k + 1], color="#16a085", lw=2.4, zorder=7)
        px, py = traj["x"][k], traj["y"][k]
        hh = traj["heading"][k]
        L, W = 4.5, 1.9
        corners = np.array(
            [[-L / 2, -W / 2], [L / 2, -W / 2], [L / 2, W / 2], [-L / 2, W / 2], [-L / 2, -W / 2]]
        )
        R = np.array([[np.cos(hh), -np.sin(hh)], [np.sin(hh), np.cos(hh)]])
        box = corners @ R.T + np.array([px, py])
        ax.plot(box[:, 0], box[:, 1], color="#2980b9", lw=2.0, zorder=8)
        ax.plot(start[0], start[1], "o", color="#27ae60", ms=9, mec="black", mew=1, zorder=8)
        ax.plot(goal[0], goal[1], "o", color="#2980b9", ms=9, mec="black", mew=1, zorder=8)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_title(f"t = {traj['t'][k]:.1f} s", fontsize=12)
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(
            fig.canvas.get_width_height()[::-1] + (3,)
        )
        frames.append(buf)
        plt.close(fig)

    from PIL import Image

    gif = out or (OUT / "trace.gif")
    Image.fromarray(frames[0]).save(
        gif,
        save_all=True,
        append_images=[Image.fromarray(f) for f in frames[1:]],
        duration=100,
        loop=0,
    )
    print(f"[demo] gif -> {gif}")


# ── Fig C: scalability summary ────────────────────────────────────────────


def fig_scalability(n_maps=6, seeds=None, out=None):
    if seeds is None:
        seeds = list(range(100, 100 + n_maps))
    rows = []
    rng = np.random.default_rng(3)
    for seed in seeds:
        hd = build_map(seed=seed)
        lanes = [l for l in hd.lanes if l.startswith("e")]
        router = CachedRouter(hd)
        ok = 0
        lens = []
        for _ in range(10):
            a, b = rng.choice(lanes, 2, replace=False)
            cla = get_lane_centerline(hd.lanes[a])
            clb = get_lane_centerline(hd.lanes[b])
            if cla is None or clb is None:
                continue
            route = router.plan(cla[0], clb[len(clb) // 2])
            if route.path is not None and len(route.path) >= 3 and not route.is_empty:
                ok += 1
                lens.append(route.total_cost)
        rows.append(
            {
                "seed": seed,
                "lanes": len(hd.lanes),
                "junctions": len(hd.junctions or {}),
                "routes_ok": ok,
                "routes_total": 10,
                "success": ok / 10.0,
                "mean_len_m": float(np.mean(lens)) if lens else 0.0,
            }
        )
        print(f"[demo] figC seed {seed}: success {ok}/10", flush=True)

    csv_path = OUT / "stats.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[demo] stats -> {csv_path}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].bar([r["seed"] for r in rows], [r["success"] * 100 for r in rows], color="#2980b9")
    axes[0].axhline(90, color="#27ae60", ls="--", lw=1.2, label="target 90%")
    axes[0].set_ylim(0, 105)
    axes[0].set_ylabel("routing success (%)")
    axes[0].set_xlabel("map seed")
    axes[0].set_title("Router success on generated maps")
    axes[0].legend()
    axes[1].bar([r["seed"] for r in rows], [r["mean_len_m"] for r in rows], color="#e67e22")
    axes[1].set_ylabel("mean route length (m)")
    axes[1].set_xlabel("map seed")
    axes[1].set_title("Route length")
    fig.suptitle(
        "Scalable closed-loop evaluation: batch route planning on generated maps",
        fontsize=13,
        y=1.02,
    )
    fig.tight_layout()
    path_out = out or (OUT / "figC_scalability.png")
    fig.savefig(path_out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[demo] figC -> {path_out}")
    return rows


# ── main ──────────────────────────────────────────────────────────────────


def main():
    os.chdir(REPO)
    print("[demo] Fig A: route-planning grid")
    fig_route_grid(seeds=(0, 1, 2, 3, 4, 5))
    print("[demo] Fig B: closed-loop driving")
    fig_closed_loop(seed=2)
    print("[demo] GIF: ego driving animation")
    make_trace_gif(seed=2)
    print("[demo] Fig C: scalability")
    fig_scalability(n_maps=6)
    print(f"[demo] done -> {OUT}")


if __name__ == "__main__":
    main()
