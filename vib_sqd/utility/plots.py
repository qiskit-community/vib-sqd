# This code is a Qiskit project.
#
# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Generic, data-agnostic matplotlib helpers.

These know nothing about vibrational structure or SQD -- they just draw the
common figure primitives the analysis plots are built from: grouped bar charts,
labelled matrix heatmaps, log-scale gap-vs-x curves, and overlaid probability
distributions.  ``vib_sqd.analysis.plots`` composes these into the
report figures.

Every helper takes an optional ``ax`` so callers can build multi-panel figures,
and returns the ``Axes`` it drew on.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")  # safe default for headless / script use
import matplotlib.pyplot as plt
import numpy as np


def grouped_bar(
    labels: Sequence[str],
    series: Mapping[str, Sequence[float]],
    *,
    ax=None,
    colors: Mapping[str, str] | None = None,
    ylabel: str = "",
    title: str = "",
    rotation: float = 60,
    fontsize: int = 8,
):
    """Grouped bar chart: one bar cluster per ``label``, one bar per ``series``.

    Used for exact/noiseless/noisy overlays and per-config |c|^2 comparisons.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 4.6))
    names = list(series)
    n = len(names)
    x = np.arange(len(labels))
    width = 0.8 / max(n, 1)
    for i, name in enumerate(names):
        off = (i - (n - 1) / 2) * width
        kw = {"label": name}
        if colors and name in colors:
            kw["color"] = colors[name]
        ax.bar(x + off, np.asarray(series[name], dtype=float), width, **kw)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=rotation, ha="right", fontsize=fontsize)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend(fontsize=fontsize)
    return ax


def heatmap_matrix(
    mat,
    labels: Sequence[str],
    *,
    ax=None,
    cmap: str = "RdBu",
    vmin: float = -1.0,
    vmax: float = 1.0,
    title: str = "",
    annotate: bool = True,
    annot_fmt: str = "{:.2f}",
    annot_atol: float = 1e-6,
    colorbar: bool = True,
):
    """Square matrix as a labelled heatmap (used for correlator generators).

    ``mat`` should already be real (pass ``.real`` or ``.imag`` as appropriate).
    Nonzero entries are annotated by default.
    """
    mat = np.asarray(mat)
    if ax is None:
        _, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    if annotate:
        for (r, c), v in np.ndenumerate(mat):
            if abs(v) > annot_atol:
                ax.text(c, r, annot_fmt.format(v), ha="center", va="center", fontsize=8)
    if title:
        ax.set_title(title)
    if colorbar:
        ax.figure.colorbar(im, ax=ax, fraction=0.046)
    return ax


def log_gap_vs_x(
    x: Sequence[float],
    curves: Mapping[str, Sequence[float]],
    *,
    ax=None,
    styles: Mapping[str, str] | None = None,
    colors: Mapping[str, str] | None = None,
    xlabel: str = "",
    ylabel: str = r"|gap to exact| (cm$^{-1}$)",
    title: str = "",
    hline: float | None = 1.0,
    hline_label: str = r"1 cm$^{-1}$",
    logx: bool = False,
    fontsize: int = 8,
):
    """Log-y gap-vs-x curves (convergence, few-step recovery, shot count).

    ``curves`` maps a series name to its y-values (absolute value taken).
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4.6))
    for name, ys in curves.items():
        style = (styles or {}).get(name, "o-")
        kw = {"label": name, "lw": 2}
        if colors and name in colors:
            kw["color"] = colors[name]
        ax.plot(x, np.abs(np.asarray(ys, dtype=float)), style, **kw)
    if hline is not None:
        ax.axhline(hline, ls=":", c="gray", alpha=0.7, label=hline_label)
    ax.set_yscale("log")
    if logx:
        ax.set_xscale("log")
    if xlabel:
        ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=fontsize)
    return ax


def overlay_distribution(
    labels: Sequence[str],
    series: Mapping[str, Sequence[float]],
    *,
    ax=None,
    colors: Mapping[str, str] | None = None,
    ylabel: str = r"probability $|c|^2$",
    xlabel: str = "configuration",
    title: str = "",
    rotation: float = 60,
    fontsize: int = 8,
    logy: bool = False,
    floor: float = 1e-4,
):
    """Overlay several |c|^2 distributions over a shared config axis.

    A thin wrapper over :func:`grouped_bar` with distribution-oriented defaults.
    ``logy`` puts the y-axis on a log scale with lower bound ``floor`` so the
    small-but-nonzero tail populations are visible (bars below ``floor``, incl.
    exact zeros, do not render).
    """
    ax = grouped_bar(
        labels,
        series,
        ax=ax,
        colors=colors,
        ylabel=ylabel,
        title=title,
        rotation=rotation,
        fontsize=fontsize,
    )
    if xlabel:
        ax.set_xlabel(xlabel)
    if logy:
        ax.set_yscale("log")
        ax.set_ylim(floor, 1.3)
    return ax
