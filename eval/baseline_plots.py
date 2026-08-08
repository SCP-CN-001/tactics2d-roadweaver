#!/usr/bin/env python3
# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Baseline evaluation figures: per-metric PNGs + combined grid."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

OUT = REPO / "analysis" / "baseline_eval"

# ── Typography ─────────────────────────────────────────────────────────────
# Times New Roman first (metrically identical to Liberation Serif); this
# machine renders Liberation Serif since Times is not installed.
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Liberation Serif", "Nimbus Roman", "DejaVu Serif"]
plt.rcParams["mathtext.fontset"] = "stix"

# ── Baseline identity (fixed across panels: color follows the entity) ──────
# Categorical palette from the dataviz reference; RoadWeaver = blue (slot 1).
ORDER = ("RoadWeaver", "MetaDrive", "RoadGen", "HDMapGen")
# Error band width = BAND_SE_SCALE × std/√n (tighter than full ±std; 1.0 = ±SE).
BAND_SE_SCALE = 1.5
BASELINES = {
    "RoadWeaver": {"color": "#2a78d6", "marker": "o", "note": "GPU", "lw": 2.4},
    "MetaDrive": {"color": "#1baf7a", "marker": "s", "note": "CPU", "lw": 2.0},
    "RoadGen": {"color": "#eda100", "marker": "^", "note": "CPU", "lw": 2.0},
    "HDMapGen": {"color": "#008300", "marker": "D", "note": "CPU", "lw": 2.0},
}

# ── Data loading ────────────────────────────────────────────────────────────


MAX_BIN = 70  # fair-comparison range: data plotted only up to 70 nodes


def load_bins(name: str) -> dict:
    """Load ``runtimes/{name}_eval/all_metrics_by_bin.csv`` as {metric: [(x, m, s, n)]}."""
    path = REPO / "runtimes" / f"{name}_eval" / "all_metrics_by_bin.csv"
    if not path.exists():  # baseline not re-run yet
        return {}
    rows = list(csv.DictReader(open(path)))
    out: dict[str, list] = {}
    for r in rows:
        b = int(float(r["node_bin"]))
        n = int(float(r["n_maps"]))
        if n < 2:  # match the "—" convention in baseline-eval.md
            continue
        for k, v in r.items():
            if not k.endswith("_mean") or v in (None, ""):
                continue
            base = k[:-5]
            std_key = f"{base}_std"
            m = float(v)
            s = float(r[std_key]) if r.get(std_key) not in (None, "") else float("nan")
            out.setdefault(base, []).append((b, m, s, n))
    for k in out:
        out[k].sort(key=lambda t: t[0])
    # Fair-comparison range: plot data only up to 70 nodes (axis pads to 75).
    for k in out:
        out[k] = [t for t in out[k] if t[0] <= MAX_BIN]
    return out


BINS = {b: load_bins(b) for b in ("roadweaver", "metadrive", "roadgen", "hdmapgen")}


# ── Plotting helpers ────────────────────────────────────────────────────────


def series(metric: str, baseline: str) -> tuple[list, list, list, list]:
    """Return (x, mean, std, n_maps) for one baseline+metric, filtered bins."""
    xs, ms, ss, ns = [], [], [], []
    for x, m, s, n in BINS[baseline.lower()].get(metric, []):
        if np.isnan(s):
            continue
        xs.append(x)
        ms.append(m)
        ss.append(s)
        ns.append(n)
    return xs, ms, ss, ns


def plot_line(
    ax,
    metric: str,
    baseline: str,
    min_val: float | None = None,
    band_cv_limit: float | None = None,
    **kw,
):
    """Line + shaded ±std band, using each baseline's fixed style.

    ``min_val``: drop per-bin points below this value (e.g. sub-10⁻² "noise").
    ``band_cv_limit``: skip the shaded error band when the baseline's median
    CV (std/mean) exceeds this (huge bands like RoadGen's 454±287 s are ugly).
    """
    xs, ms, ss, ns = series(metric, baseline)
    if not xs:
        return
    if min_val is not None:
        keep = [i for i, m in enumerate(ms) if m >= min_val]
        xs = [xs[i] for i in keep]
        ms = [ms[i] for i in keep]
        ss = [ss[i] for i in keep]
        ns = [ns[i] for i in keep]
        if not xs:
            return
    draw_band = True
    if band_cv_limit is not None:
        cv = [s / m if m > 0 else 0.0 for m, s in zip(ms, ss)]
        if np.median(cv) > band_cv_limit:
            draw_band = False
    c = BASELINES[baseline]["color"]
    ax.plot(
        xs,
        ms,
        color=c,
        lw=BASELINES[baseline]["lw"],
        marker=BASELINES[baseline]["marker"],
        ms=6,
        **kw,
    )
    if draw_band:
        # SE band scaled by BAND_SE_SCALE (std/√n × scale): tighter than ±std.
        se = [BAND_SE_SCALE * s / np.sqrt(max(n, 1)) for s, n in zip(ss, ns)]
        ax.fill_between(
            xs,
            [m - e for m, e in zip(ms, se)],
            [m + e for m, e in zip(ms, se)],
            color=c,
            alpha=0.15,
            lw=0,
            zorder=0,
        )


def style_ax(ax, *, log: bool = False, ylim=None, xticks=None, labelsize: int = 10):
    """Apply the light-surface chart chrome."""
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#c3c2b7")
    ax.tick_params(colors="#52514e", labelsize=labelsize)
    ax.yaxis.grid(True, color="#e1e0d9", lw=0.6, alpha=0.6, zorder=0)
    if log:
        ax.set_yscale("log")
        ax.grid(True, which="both", color="#e1e0d9", lw=0.4, alpha=0.4, zorder=0)
    if ylim:
        ax.set_ylim(*ylim)
    if xticks is not None:
        ax.set_xticks(xticks)


def draw_hdmap_cap(ax):
    """Vertical marker: the three baselines' data ends at ~60 (RoadWeaver to 80)."""
    ax.axvline(60.5, color="#008300", ls="--", lw=1.0, alpha=0.55)
    ax.text(
        61.8,
        0.5,
        "other baselines stop at 60",
        rotation=90,
        va="center",
        ha="left",
        transform=ax.get_xaxis_transform(),
        color="#008300",
        fontsize=8,
        alpha=0.8,
    )


def make_handles():
    return [
        plt.Line2D(
            [],
            [],
            color=BASELINES[b]["color"],
            marker=BASELINES[b]["marker"],
            ms=7,
            lw=BASELINES[b]["lw"],
            label=f"{b} ({BASELINES[b]['note']})",
        )
        for b in ORDER
    ]


# ── Individual figures ──────────────────────────────────────────────────────


def save_single(
    metric: str,
    fname: str,
    ylabel: str,
    *,
    log: bool = False,
    ylim=None,
    xlim=None,
    cap: bool = False,
    special=None,
    xticks=None,
    figsize=(6.3, 4.4),
    fontsize: int = 11,
    min_val: float | None = None,
    band_cv_limit: float | None = None,
    legend: bool = True,
    legend_bbox=None,
    legend_ncol: int = 1,
    legend_fontsize: int | None = None,
):
    """Render one metric as a standalone PNG."""
    fig, ax = plt.subplots(figsize=figsize)
    if special is not None:
        special(ax, min_val=min_val, band_cv_limit=band_cv_limit)
    else:
        for b in ORDER:
            plot_line(ax, metric, b, min_val=min_val, band_cv_limit=band_cv_limit)
    if ylim is None and not log:
        ylim = compute_ylim(metric)
    style_ax(ax, log=log, ylim=ylim, xticks=xticks, labelsize=fontsize)
    if xlim:
        ax.set_xlim(*xlim)
    if cap:
        draw_hdmap_cap(ax)
    ax.set_xlabel("Number of intersection nodes", fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    lf = legend_fontsize if legend_fontsize is not None else fontsize - 1
    if legend:
        if legend_bbox is not None:
            ax.legend(
                handles=make_handles(),
                loc="upper left",
                bbox_to_anchor=legend_bbox,
                frameon=False,
                fontsize=lf,
                ncol=legend_ncol,
                labelspacing=0.15,
                handlelength=1.0,
                handletextpad=0.4,
                borderaxespad=0.3,
                columnspacing=0.8,
            )
        else:
            ax.legend(handles=make_handles(), loc="best", frameon=False, fontsize=lf, ncol=1)
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved {OUT / fname}")
    plt.close(fig)


def _cycle_special(ax, **kw):
    """Cycle ratio from the binned CSVs; RoadGen is a tree → 0 (not measured)."""
    for b in ("RoadWeaver", "MetaDrive", "HDMapGen"):
        xs, ms, ss, ns = series("cycle_ratio", b)
        if not xs:
            continue
        ax.plot(
            xs,
            ms,
            color=BASELINES[b]["color"],
            lw=BASELINES[b]["lw"],
            marker=BASELINES[b]["marker"],
            ms=6,
        )
        ax.fill_between(
            xs,
            [m - s for m, s in zip(ms, ss)],
            [m + s for m, s in zip(ms, ss)],
            color=BASELINES[b]["color"],
            alpha=0.15,
            lw=0,
            zorder=0,
        )
    # RoadGen: tree WidgetGraph structurally has no cycles → flat 0.
    xs, _, _, _ = series("dead_end_ratio", "RoadGen")
    if xs:
        ax.plot(
            xs,
            [0.0] * len(xs),
            color=BASELINES["RoadGen"]["color"],
            marker=BASELINES["RoadGen"]["marker"],
            ms=6,
            lw=1.2,
            ls="--",
        )


def _gpu_special(ax, **kw):
    """Per-process GPU peak; CPU baselines have no CUDA context → 0."""
    for b in ("RoadWeaver", "MetaDrive", "HDMapGen"):
        plot_line(ax, "gpu_peak_mb", b)
    # RoadGen: CPU solver, never touches CUDA → 0.
    xs, _, _, _ = series("dead_end_ratio", "RoadGen")
    if xs:
        ax.plot(
            xs,
            [0.0] * len(xs),
            color=BASELINES["RoadGen"]["color"],
            marker=BASELINES["RoadGen"]["marker"],
            ms=6,
            lw=1.2,
            ls="--",
        )
    ax.text(
        0.985,
        0.04,
        "CPU baselines: no CUDA context\n(per-process nvidia-smi)",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="#52514e",
    )


def _cpu_time_special(ax, min_val: float | None = None, band_cv_limit: float | None = None, **kw):
    """CPU-seconds ≈ gen_time_s × cpu_peak/100 per bin (from by_bin, if available).

    ``monitor_resources`` tracks peak CPU %, not an integral, so this is an
    upper-bound estimate of total CPU time — accurate for single-threaded,
    sustained-load baselines (RoadGen), approximate for short GPU bursts
    (RoadWeaver).
    """
    for b in ORDER:
        path = REPO / "runtimes" / f"{b.lower()}_eval" / "all_metrics_by_bin.csv"
        if not path.exists():
            continue
        rows = [r for r in csv.DictReader(open(path)) if int(float(r["n_maps"])) >= 2]
        xs, ms, ss, ns = [], [], [], []
        for r in rows:
            if int(float(r["node_bin"])) > MAX_BIN:
                continue
            g, c = float(r["gen_time_s_mean"]), float(r["cpu_peak_mean"])
            if g <= 0 or c <= 0:
                continue
            gs, cs = float(r["gen_time_s_std"]), float(r["cpu_peak_std"])
            ct = g * c / 100.0  # CPU-seconds
            if min_val is not None and ct < min_val:
                continue
            ct_std = ct * np.sqrt((gs / g) ** 2 + (cs / c) ** 2) if (gs > 0 and cs > 0) else 0.0
            xs.append(int(float(r["node_bin"])))
            ms.append(ct)
            ss.append(ct_std)
            ns.append(int(float(r["n_maps"])))
        if not xs:
            continue
        draw_band = True
        if band_cv_limit is not None:
            cv = [s / m if m > 0 else 0.0 for m, s in zip(ms, ss)]
            if np.median(cv) > band_cv_limit:
                draw_band = False
        st = BASELINES[b]
        ax.plot(xs, ms, color=st["color"], lw=st["lw"], marker=st["marker"], ms=6)
        if draw_band:
            se = [BAND_SE_SCALE * s / np.sqrt(max(n, 1)) for s, n in zip(ss, ns)]
            ax.fill_between(
                xs,
                [m - e for m, e in zip(ms, se)],
                [m + e for m, e in zip(ms, se)],
                color=st["color"],
                alpha=0.15,
                lw=0,
                zorder=0,
            )


# ── Data-driven y-limits (trim outliers; don't force 0) ─────────────────────

STRUCTURAL_ZERO = {"cycle_ratio": ["RoadGen"], "gpu_peak_mb": ["RoadGen"]}


def _ylim_values(metric: str, baseline: str) -> list:
    """Per-bin means actually plotted for one baseline+metric (incl. derived)."""
    if metric == "cpu_time_s":
        path = REPO / "runtimes" / f"{baseline.lower()}_eval" / "all_metrics_by_bin.csv"
        if not path.exists():
            return []
        out = []
        for r in csv.DictReader(open(path)):
            if int(float(r["n_maps"])) < 2:
                continue
            g, c = float(r["gen_time_s_mean"]), float(r["cpu_peak_mean"])
            if g > 0 and c > 0:
                out.append(g * c / 100.0)
        return out
    if metric in STRUCTURAL_ZERO and baseline in STRUCTURAL_ZERO[metric]:
        xs, _, _, _ = series("dead_end_ratio", baseline)
        return [0.0] * len(xs)
    _x, ms, _s, _n = series(metric, baseline)
    return ms


def compute_ylim(metric: str, pad: float = 0.08) -> tuple | None:
    """Robust y-range from plotted per-bin means.

    - Drops non-zero "noise" values near 0 (< 5 % of the max, e.g. HDMapGen's
      tiny cycle-ratio bins) but keeps genuine 0 baselines (RoadGen cycle, CPU
      GPU) so the axis still starts at 0 where 0 is meaningful.
    - Trims only the top outliers (high-end spikes), never the bottom, so real
      low values (e.g. HDMapGen's LCC fragmentation dip at ~0.25) stay visible.
    """
    vals = np.asarray([v for b in ORDER for v in _ylim_values(metric, b)], dtype=float)
    if len(vals) < 2:
        return None
    vals = np.sort(vals)
    if vals.max() > 0:
        thresh = 0.05 * vals.max()
        vals = vals[(vals >= thresh) | (vals == 0)]
    k = max(1, len(vals) // 10)
    if len(vals) > k:
        vals = vals[:-k]
    if len(vals) == 0:
        return None
    lo, hi = float(vals.min()), float(vals.max())
    span = max(hi - lo, 1e-9)
    return (max(lo - pad * span, 0.0), hi + pad * span)


# ── Entry point ─────────────────────────────────────────────────────────────


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--only",
        default="",
        help="comma-separated figure names to generate " "(e.g. gen_time,cpu_time); default = all",
    )
    args = ap.parse_args()
    only = {s.strip() for s in args.only.split(",") if s.strip()}
    FIG = {
        "gen_time_s": "gen_time",
        "cycle_ratio": "cycle_ratio",
        "dead_end_ratio": "dead_end_ratio",
        "lcc": "lcc",
        "cpu_time_s": "cpu_time",
        "gpu_peak_mb": "gpu_mem",
        "mem_peak_mb": "mem_peak",
    }

    def want(fname):
        return (not only) or (fname in only)

    OUT.mkdir(parents=True, exist_ok=True)
    xticks = list(range(10, 71, 10))
    XLIM = (0, 75)

    if want("gen_time"):
        save_single(
            "gen_time_s",
            "gen_time.png",
            "Generation time (s, log)",
            log=True,
            ylim=(1e-2, 1e4),
            xlim=XLIM,
            cap=False,
            xticks=xticks,
            figsize=(6.0, 5.3),
            fontsize=16,
            min_val=1e-2,
            legend_bbox=(0.02, 1.0),
            legend_ncol=1,
            legend_fontsize=12,
        )
    if want("cycle_ratio"):
        save_single(
            "cycle_ratio",
            "cycle_ratio.png",
            "Cycle ratio",
            ylim=None,
            cap=True,
            special=_cycle_special,
            xticks=xticks,
        )
    if want("dead_end_ratio"):
        save_single(
            "dead_end_ratio", "dead_end_ratio.png", "Dead-end ratio", ylim=None, xticks=xticks
        )
    if want("lcc"):
        save_single("lcc", "lcc.png", "LCC", ylim=None, xticks=xticks)
    if want("gpu_mem"):
        save_single(
            "gpu_peak_mb",
            "gpu_mem.png",
            "GPU memory peak (MB)",
            ylim=None,
            special=_gpu_special,
            xticks=xticks,
        )
    if want("mem_peak"):
        save_single("mem_peak_mb", "mem_peak.png", "Memory peak (MB)", ylim=None, xticks=xticks)
    if want("cpu_time"):
        save_single(
            "cpu_time_s",
            "cpu_time.png",
            "CPU time (CPU·s, log)",
            log=True,
            ylim=(1e-3, 1e4),
            xlim=XLIM,
            cap=False,
            special=_cpu_time_special,
            xticks=xticks,
            figsize=(6.0, 5.3),
            fontsize=16,
            min_val=1e-3,
            legend=False,
        )

    # ── Combined grid for reference (only selected panels) ──
    panels = [
        ("gen_time_s", "Generation time (s, log)", True, (1e-3, 1e3), False, None),
        ("cpu_time_s", "CPU time (CPU·s, log)", True, (1e-3, 1e3), False, _cpu_time_special),
    ]
    panels = [p for p in panels if want(FIG[p[0]])]
    MIN_VAL = {"gen_time_s": 1e-2, "cpu_time_s": 1e-3}
    ncol = 2 if len(panels) <= 2 else 3
    nrow = (len(panels) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(8.0 * ncol, 8.0 * nrow))
    fig.subplots_adjust(hspace=0.45, wspace=0.12, top=0.90, bottom=0.10)
    axlist = np.atleast_1d(axes).ravel()
    for ax, (metric, ylabel, log, ylim, cap, special) in zip(axlist, panels):
        mv = MIN_VAL.get(metric)
        if special is not None:
            special(ax, min_val=mv)
        else:
            for b in ORDER:
                plot_line(ax, metric, b, min_val=mv)
        if ylim is None and not log:
            ylim = compute_ylim(metric)
        style_ax(ax, log=log, ylim=ylim, xticks=xticks, labelsize=15)
        ax.set_xlim(*XLIM)
        if cap:
            draw_hdmap_cap(ax)
        ax.set_xlabel("Number of intersection nodes", fontsize=18)
        ax.set_ylabel(ylabel, fontsize=18)
    for ax in axlist[len(panels) :]:
        ax.axis("off")
    fig.legend(
        handles=make_handles(),
        loc="upper center",
        ncol=4,
        frameon=False,
        fontsize=18,
        bbox_to_anchor=(0.5, 0.985),
    )
    fig.savefig(OUT / "combined.png", dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved {OUT / 'combined.png'}")
    plt.close(fig)


if __name__ == "__main__":
    main()
