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

"""Report-figure plotting for the MidasCpp vibrational-SQD study.

Each function takes ALREADY-COMPUTED arrays/dicts (it does not run any
experiment) and writes a figure to ``outpath``.  The experiment scripts in
``workflows/midascpp_discovery/pipeline/`` compute the data and call these; the
tutorial notebooks call them too.  Generic drawing primitives live in
``vib_sqd.utility.plots``.

Colour convention (kept consistent across figures):
    VIm-uCJ #c1121f, Vg-uCJ #7209b7, CHC #2a6f97, VLUCJ #e07a5f.
"""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from vib_sqd.utility import plots as up

ANSATZ_COLORS = {
    "VIm-uCJ": "#c1121f",
    "Vg-uCJ": "#7209b7",
    "CHC": "#2a6f97",
    "VLUCJ": "#e07a5f",
    "CHC+4dbl": "#2a6f97",
    "UVCCSD": "#6a994e",
    "uniform": "#adb5bd",
}


# --------------------------------------------------------------------------- #
# VLUCJ levers (gap + coverage, HO vs modal)
# --------------------------------------------------------------------------- #
def plot_vlucj_levers(results: dict, variants: Sequence[str], outpath):
    """results[basis]['variants'][name][metric] for basis in {ho, modal}."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    x = np.arange(len(variants))
    for ax, metric, title, ylab in [
        (
            axes[0],
            "gap",
            "Energy gap to exact (lower = better)",
            r"$E-E_{\rm exact}$ (cm$^{-1}$)",
        ),
        (
            axes[1],
            "top10_cov",
            "SQD top-10 config coverage (higher = better)",
            "top-10 configs covered",
        ),
    ]:
        ho = [results["ho"]["variants"][n][metric] for n in variants]
        mo = [results["modal"]["variants"][n][metric] for n in variants]
        ax.bar(x - 0.2, ho, 0.4, color="#2a6f97", label="HO basis")
        ax.bar(x + 0.2, mo, 0.4, color="#c1121f", label="modal basis")
        ax.set_xticks(x)
        ax.set_xticklabels(variants, rotation=30, ha="right", fontsize=8)
        ax.set_title(title)
        ax.set_ylabel(ylab)
        ax.legend()
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle("Glycine-4: VLUCJ levers vs CHC (HO vs modal basis)")
    fig.tight_layout()
    fig.savefig(outpath, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# VIm-uCJ math (generators + transfer maps)
# --------------------------------------------------------------------------- #
def plot_vimucj_math(
    Gi, Gc, thetas, w_im, w_cp, w_gv, blk_i, blk_c, phys_labels: Sequence[str], outpath
):
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 3)

    ax = fig.add_subplot(gs[0, 0])
    up.heatmap_matrix(
        np.real(Gi),
        phys_labels,
        ax=ax,
        title="(1a) VIm-uCJ generator $G_{\\rm im}$\n(real, "
        "anti-symmetric, OFF-diagonal)",
    )
    ax = fig.add_subplot(gs[0, 1])
    up.heatmap_matrix(
        np.imag(Gc),
        phys_labels,
        ax=ax,
        title="(1b) VLUCJ CP generator $\\mathrm{Im}\\,G_{\\rm cp}$\n"
        "(imaginary, DIAGONAL => pure phase)",
    )

    ax = fig.add_subplot(gs[0, 2])
    ax.axis("off")
    txt = (
        "(4) Effective 2-level "
        "$\\{|{\\rm ref}\\rangle,|{\\rm dbl}\\rangle\\}$\n\n"
        f"VIm-uCJ block $G_{{\\rm im}}$:\n{np.real(blk_i)}\n"
        "= SO(2) generator -> Rabi rotation\nexp$(\\theta G)$ moves amplitude\n\n"
        f"VLUCJ CP block $G_{{\\rm cp}}$:\n{np.round(np.imag(blk_c), 3)}$\\,i$\n"
        "= diagonal -> relative phase only\nreference amplitude unchanged"
    )
    ax.text(0.0, 1.0, txt, va="top", ha="left", family="monospace", fontsize=8.5)

    dbl_i = list(phys_labels).index("11")
    for col, (w, xlab, title) in enumerate(
        [
            (
                w_im,
                "$\\theta$",
                "(2a) VIm-uCJ transfer: weight vs $\\theta$\n"
                "(builds amplitude on the double '11')",
            ),
            (
                w_cp,
                "$\\phi$",
                "(2b) VLUCJ CP transfer: weight vs $\\phi$\n"
                "(reference weight flat = inert)",
            ),
            (
                w_gv,
                "$\\theta$",
                "(2c) Intramode Givens transfer\n"
                "(single-mode only: '00'->'10', never '11')",
            ),
        ]
    ):
        ax = fig.add_subplot(gs[1, col])
        for c in range(len(phys_labels)):
            ax.plot(
                thetas, w[:, c], label=phys_labels[c], lw=2.4 if c == dbl_i else 1.0
            )
        ax.set_title(title)
        ax.set_xlabel(xlab)
        ax.set_ylabel(r"$|\langle c|U|{\rm ref}\rangle|^2$")
        if col == 1:
            ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=7, title="config")

    fig.suptitle(
        "What VIm-uCJ does: an anti-Hermitian correlator ROTATES paired "
        "amplitude; the diagonal CP only PHASES",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(outpath, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Vg-uCJ math (rotation + phase generators, complex disk, interference toy)
# --------------------------------------------------------------------------- #
def plot_vgucj_math(
    G_rot,
    G_ph,
    thetas,
    c_im,
    TT,
    PP,
    c_g,
    f_im,
    f_g,
    phys_labels: Sequence[str],
    outpath,
):
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 3)

    ax = fig.add_subplot(gs[0, 0])
    up.heatmap_matrix(
        np.real(G_rot),
        phys_labels,
        ax=ax,
        title="(1a) $G_{\\rm rot}$ (imaginary rotation)\nreal, "
        "antisymmetric, OFF-diagonal",
    )
    ax = fig.add_subplot(gs[0, 1])
    up.heatmap_matrix(
        np.imag(G_ph),
        phys_labels,
        ax=ax,
        title="(1b) $\\mathrm{Im}\\,G_{\\rm ph}$ (real/phase part)\n"
        "diagonal, nonzero on the DOUBLE '11'",
    )

    ax = fig.add_subplot(gs[0, 2])
    ax.scatter(
        c_g.real.ravel(),
        c_g.imag.ravel(),
        s=6,
        c="#7209b7",
        alpha=0.35,
        label="Vg-uCJ (complex disk)",
    )
    ax.plot(
        c_im.real, c_im.imag, "-", color="#c1121f", lw=3, label="VIm-uCJ (real segment)"
    )
    ax.set_xlabel("Re $c_{\\rm dbl}$")
    ax.set_ylabel("Im $c_{\\rm dbl}$")
    ax.set_title(
        "(2) reachable amplitude on the double\nVIm-uCJ: real line;  "
        "Vg-uCJ: full disk"
    )
    ax.axhline(0, c="k", lw=0.5)
    ax.axvline(0, c="k", lw=0.5)
    ax.set_aspect("equal")
    ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[1, 0])
    im = ax.imshow(
        np.abs(c_g),
        extent=[thetas.min(), thetas.max(), 0, 2 * np.pi],
        origin="lower",
        aspect="auto",
        cmap="viridis",
    )
    ax.set_title("(2b) Vg-uCJ $|c_{\\rm dbl}|(\\theta,\\phi)$")
    ax.set_xlabel("$\\theta$ (rotation)")
    ax.set_ylabel("$\\phi$ (phase)")
    fig.colorbar(im, ax=ax, fraction=0.046)

    ax = fig.add_subplot(gs[1, 1])
    im = ax.imshow(
        np.angle(c_g),
        extent=[thetas.min(), thetas.max(), 0, 2 * np.pi],
        origin="lower",
        aspect="auto",
        cmap="twilight",
    )
    ax.set_title(
        "(2c) Vg-uCJ $\\arg\\,c_{\\rm dbl}(\\theta,\\phi)$\n(phi tunes the "
        "phase VIm-uCJ cannot)"
    )
    ax.set_xlabel("$\\theta$ (rotation)")
    ax.set_ylabel("$\\phi$ (phase)")
    fig.colorbar(im, ax=ax, fraction=0.046)

    ax = fig.add_subplot(gs[1, 2])
    ax.bar(
        ["VIm-uCJ\n(phase=0)", "Vg-uCJ\n(phase free)"],
        [f_im, f_g],
        color=["#c1121f", "#7209b7"],
    )
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("best fidelity to target")
    ax.set_title(
        "(3) two-double interference toy\n(target needs opposite-sign " "amplitudes)"
    )
    for i, v in enumerate([f_im, f_g]):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=10)

    fig.suptitle(
        "What Vg-uCJ adds: the imaginary part TRANSFERS amplitude; the "
        "real/phase part CHOOSES its phase (coherent interference)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(outpath, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Sampled coverage (top-K configs and weight vs shots)
# --------------------------------------------------------------------------- #
def plot_sampled_coverage(shots, rec_curves, wt_curves, K, top_w, outpath):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for name, rec in rec_curves.items():
        axes[0].plot(shots, rec, "o-", color=ANSATZ_COLORS.get(name), label=name, lw=2)
    axes[0].axhline(K, ls="--", c="k", alpha=0.4, label=f"all {K}")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("shots $N$")
    axes[0].set_ylabel(f"expected top-{K} configs recovered")
    axes[0].set_title(f"SQD coverage: exact top-{K} configs found vs shots")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    for name, wt in wt_curves.items():
        axes[1].plot(shots, wt, "o-", color=ANSATZ_COLORS.get(name), label=name, lw=2)
    axes[1].axhline(
        top_w, ls="--", c="k", alpha=0.4, label=f"top-{K} weight={top_w:.3f}"
    )
    axes[1].set_xscale("log")
    axes[1].set_xlabel("shots $N$")
    axes[1].set_ylabel("expected exact-weight recovered")
    axes[1].set_title("SQD coverage: exact ground-state weight captured vs shots")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    fig.suptitle(
        "Glycine-4 modal: finite-shot SQD coverage per ansatz "
        "(the metric that governs projection)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(outpath, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# UCJ convergence (gap vs optimizer steps, per system x basis)
# --------------------------------------------------------------------------- #
def plot_ucj_convergence(results: dict, budgets, outpath):
    systems = list(results.keys())
    fig, axes = plt.subplots(
        len(systems), 2, figsize=(12, 5 * len(systems)), squeeze=False
    )
    styles = {"VIm-uCJ": "o-", "Vg-uCJ": "s--"}
    for r, label in enumerate(systems):
        for c, basis in enumerate(("modal", "ho")):
            ax = axes[r][c]
            d = results[label][basis]
            curves = {name: d[name] for name in ("VIm-uCJ", "Vg-uCJ") if name in d}
            legend = {
                n: f"{n} (init gap {np.asarray(v)[0]:+.1f})" for n, v in curves.items()
            }
            up.log_gap_vs_x(
                budgets,
                {legend[n]: curves[n] for n in curves},
                ax=ax,
                styles={legend[n]: styles.get(n, "o-") for n in curves},
                colors={legend[n]: ANSATZ_COLORS.get(n) for n in curves},
                xlabel="optimizer steps",
                title=f"{label} / {basis} basis (CCSD warm start)",
            )
    fig.suptitle(
        "VIm-uCJ & Vg-uCJ convergence from a bounded CCSD/VCC-t2 warm "
        "start (few-step, SQD-style)",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(outpath, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# SQD few-step recovered gap vs steps
# --------------------------------------------------------------------------- #
def plot_sqd_fewstep(results: dict, steps, outpath):
    systems = list(results.keys())
    fig, axes = plt.subplots(
        len(systems), 2, figsize=(13, 5 * len(systems)), squeeze=False
    )
    for r, label in enumerate(systems):
        for c, basis in enumerate(("modal", "ho")):
            ax = axes[r][c]
            V = results[label][basis]["variants"]
            curves = {}
            for name in V:
                curves[name] = [
                    V[name][str(s) if str(s) in V[name] else s]["rec_gap"]
                    for s in steps
                ]
            up.log_gap_vs_x(
                steps,
                curves,
                ax=ax,
                styles={n: "o-" for n in curves},
                colors={n: ANSATZ_COLORS.get(n) for n in curves},
                xlabel="optimizer steps before sampling",
                ylabel=r"|recovered gap| (cm$^{-1}$)",
                title=f"{label} / {basis}: SQD recovered gap (sample+grow subspace)",
            )
            ax.set_xticks(steps)
    fig.suptitle(
        "SQD few-step test: recovered energy after 0-5 steps + iterative "
        "subspace growth (noiseless)",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(outpath, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Noisy benchmark distributions
# --------------------------------------------------------------------------- #
def plot_noisy_overlay(
    occs,
    p_exact,
    dist,
    name,
    frac,
    fewsteps,
    outpath,
    k=16,
    source="FakeSherbrooke",
    logy=True,
    floor=1e-4,
):
    order = np.argsort(p_exact)[::-1][:k]
    labels = ["".join(map(str, occs[i])) for i in order]
    series = {
        "exact": p_exact[order],
        "noiseless circuit": dist["noiseless"][order],
        "noisy (raw, one-hot mass)": dist["noisy_raw"][order],
        "noisy + one-hot post-sel": dist["noisy_ps"][order],
    }
    colors = {
        "exact": "#000000",
        "noiseless circuit": "#2a6f97",
        "noisy (raw, one-hot mass)": "#f4a261",
        "noisy + one-hot post-sel": "#c1121f",
    }
    fig, ax = plt.subplots(figsize=(13, 5))
    up.overlay_distribution(
        labels,
        series,
        ax=ax,
        colors=colors,
        logy=logy,
        floor=floor,
        ylabel=r"probability $|c|^2$" + (" (log)" if logy else ""),
        xlabel="configuration (occupation, top-16 by exact weight)",
        title=f"{name} on glycine-4 modal ({source}, {fewsteps}-step warm start"
        + (", log scale" if logy else "")
        + f"): one-hot post-selection retains {frac:.0%} of shots",
    )
    fig.tight_layout()
    fig.savefig(outpath, dpi=130)
    plt.close(fig)


def plot_noisy_compare(
    occs,
    p_exact,
    dists,
    results,
    names,
    outpath,
    k=14,
    source="FakeSherbrooke",
    logy=True,
    floor=1e-4,
):
    """Per-ansatz noisy post-selected |c|^2 vs exact.

    logy=True (default) uses a log y-axis: the reference config dominates the
    linear scale and makes the small-but-nonzero TAIL populations (the configs
    SQD actually recovers from) look like zero. On log scale they are visible,
    which is the informative view. ``floor`` is the log-axis lower bound; bars
    below it (incl. genuinely-unsampled configs, which are exactly 0) simply do
    not show -- an honest depiction of "not present in the sample".
    """
    order = np.argsort(p_exact)[::-1][:k]
    labels = ["".join(map(str, occs[i])) for i in order]
    x = np.arange(k)
    fig, axes = plt.subplots(1, len(names), figsize=(16, 4.6), sharey=True)
    if len(names) == 1:
        axes = [axes]
    for ax, name in zip(axes, names):
        ax.bar(x - 0.22, p_exact[order], 0.44, label="exact", color="#000000")
        ax.bar(
            x + 0.22,
            dists[name]["noisy_ps"][order],
            0.44,
            label="noisy+PS",
            color=ANSATZ_COLORS.get(name),
        )
        r = results["variants"][name]
        ax.set_title(
            f"{name}\nretained {r['onehot_retained_frac']:.0%}, "
            f"noisy+PS+SQD gap {r['gap_noisy_ps_rec']:.2f} cm$^{{-1}}$",
            fontsize=10,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
        ax.set_xlabel("config")
        if logy:
            ax.set_yscale("log")
            ax.set_ylim(floor, 1.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel(r"probability $|c|^2$" + (" (log)" if logy else ""))
    fig.suptitle(
        "Noisy post-selected distribution vs exact"
        + (" (log scale)" if logy else "")
        + ": "
        + " vs ".join(names)
        + f" (glycine-4 modal, {source})",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(outpath, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 32-qubit glycine SQD (naive-discard vs configuration recovery, + convergence)
# --------------------------------------------------------------------------- #
def plot_glycine8_recovery_convergence(
    traces, outpath, reference_cm=None, sd_target_cm=None, title_extra=""
):
    """Recovered energy vs SQD configuration-recovery iteration, per ansatz.
    ``traces`` maps ansatz -> list of per-iteration energies (cm^-1). Optional
    horizontal lines: reference-config energy and the reference+S/D target."""
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for name, tr in traces.items():
        ax.plot(
            range(1, len(tr) + 1),
            tr,
            "o-",
            lw=2,
            color=ANSATZ_COLORS.get(name),
            label=name,
        )
    if reference_cm is not None:
        ax.axhline(
            reference_cm,
            ls="--",
            c="gray",
            alpha=0.8,
            label=f"reference config ({reference_cm:.1f})",
        )
    if sd_target_cm is not None:
        ax.axhline(
            sd_target_cm,
            ls=":",
            c="k",
            alpha=0.8,
            label=f"ref+S/D ground ({sd_target_cm:.1f})",
        )
    ax.set_xlabel("SQD configuration-recovery iteration")
    ax.set_ylabel(r"recovered energy (cm$^{-1}$)")
    ax.set_title("32-qubit glycine: SQD recovery convergence" + title_extra)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outpath, dpi=130)
    plt.close(fig)


def plot_glycine8_naive_vs_recovery(
    naive, recovered, outpath, reference_cm=None, sd_target_cm=None
):
    """Grouped bar: naive one-hot-discard vs configuration-recovery energy per
    ansatz (log y -- naive spans thousands). ``naive``/``recovered`` map
    ansatz -> energy cm^-1."""
    names = list(recovered)
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(
        x - 0.2,
        [naive[n] for n in names],
        0.4,
        label="naive (discard)",
        color="#adb5bd",
    )
    ax.bar(
        x + 0.2,
        [recovered[n] for n in names],
        0.4,
        label="recovery (repair)",
        color="#c1121f",
    )
    if sd_target_cm is not None:
        ax.axhline(
            sd_target_cm,
            ls=":",
            c="k",
            alpha=0.8,
            label=f"ref+S/D ground ({sd_target_cm:.1f})",
        )
    if reference_cm is not None:
        ax.axhline(
            reference_cm,
            ls="--",
            c="gray",
            alpha=0.7,
            label=f"reference ({reference_cm:.1f})",
        )
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel(r"energy (cm$^{-1}$, log)")
    ax.set_title("32-qubit glycine: configuration recovery rescues the collapsed run")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outpath, dpi=130)
    plt.close(fig)


def plot_glycine8_depth(depth, outpath):
    """Bar chart of transpiled 2q (CZ) count + depth per ansatz at 32 qubits.
    ``depth`` maps ansatz -> (l3_cz, l3_depth)."""
    names = list(depth)
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(10, 5))
    cz = [depth[n][0] for n in names]
    dp = [depth[n][1] for n in names]
    ax.bar(x - 0.2, cz, 0.4, label="2q (CZ) count", color="#2a6f97")
    ax.bar(x + 0.2, dp, 0.4, label="circuit depth", color="#e07a5f")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("count")
    ax.set_title("32-qubit glycine: transpiled circuit cost (Heron L3)")
    for i, (c, d) in enumerate(zip(cz, dp)):
        ax.text(i - 0.2, c, str(c), ha="center", va="bottom", fontsize=7)
        ax.text(i + 0.2, d, str(d), ha="center", va="bottom", fontsize=7)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outpath, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 64-qubit glycine: two configs (16x4 vs 8x8) + 5M-vs-100k shot study
# --------------------------------------------------------------------------- #
def plot_glycine64_config_compare(rec16, rec8, outpath, ref16_cm=None, ref8_cm=None):
    """Grouped bar: recovered energy per ansatz for the two 64-qubit configs
    (16 modes x 4 modal vs 8 modes x 8 modal). Both collapse to 0% raw one-hot
    and both recover to the physical energy of their OWN active space -- the two
    clusters are different mode sets, not success-vs-failure. ``rec16``/``rec8``
    map ansatz -> recovered energy cm^-1; ``ref16_cm``/``ref8_cm`` are each
    space's reference-config energy (dotted lines)."""
    names = list(rec8)
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.bar(
        x - 0.2,
        [rec16[n] for n in names],
        0.4,
        label="16 modes x 4 modal (modes 12-27)",
        color="#b23a48",
    )
    ax.bar(
        x + 0.2,
        [rec8[n] for n in names],
        0.4,
        label="8 modes x 8 modal (modes 20-27)",
        color="#2a6f97",
    )
    if ref16_cm is not None:
        ax.axhline(
            ref16_cm,
            ls=":",
            c="#b23a48",
            alpha=0.9,
            label=f"16x4 reference config ({ref16_cm:.0f})",
        )
    if ref8_cm is not None:
        ax.axhline(
            ref8_cm,
            ls=":",
            c="#2a6f97",
            alpha=0.9,
            label=f"8x8 reference config ({ref8_cm:.0f})",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel(r"recovered energy (cm$^{-1}$)")
    ax.set_title(
        "64-qubit glycine: two active spaces, both recovered to their own physical E\n"
        "(both 64 qubits, both 0.000% raw one-hot; bars sit near each space's reference)"
    )
    for i, n in enumerate(names):
        ax.text(
            i - 0.2, rec16[n], f"{rec16[n]:.0f}", ha="center", va="bottom", fontsize=7
        )
        ax.text(
            i + 0.2, rec8[n], f"{rec8[n]:.0f}", ha="center", va="bottom", fontsize=7
        )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outpath, dpi=130)
    plt.close(fig)


def plot_glycine64_shots_study(
    shot_totals, valid_counts, rec_energies, outpath, sd_target_cm=None
):
    """Twin-axis: valid one-hot COUNT (bars, left) and recovered energy (line,
    right) vs total shots for the VIm-uCJ 16x4 run. Shows that 50x more shots
    (100k -> 5M) surfaces only a handful of valid samples and does NOT move the
    recovered energy -- the 16-mode collapse is not a statistics problem.
    All three sequences are aligned to ``shot_totals``."""
    x = np.arange(len(shot_totals))
    fig, ax = plt.subplots(figsize=(9, 5.2))
    bars = ax.bar(x, valid_counts, 0.5, color="#457b9d", label="valid one-hot samples")
    for i, v in enumerate(valid_counts):
        ax.text(i, v, str(v), ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{s/1e6:g}M" if s >= 1e6 else f"{s/1e3:g}k" for s in shot_totals]
    )
    ax.set_xlabel("total shots (VIm-uCJ, 16 modes x 4 modal, DD only)")
    ax.set_ylabel("valid one-hot samples out of total", color="#457b9d")
    ax.tick_params(axis="y", labelcolor="#457b9d")
    ax.set_ylim(0, max(valid_counts) * 1.4 + 1)

    ax2 = ax.twinx()
    ax2.plot(x, rec_energies, "o-", color="#c1121f", lw=2, label="recovered energy")
    for i, e in enumerate(rec_energies):
        ax2.text(
            i, e, f"{e:.0f}", ha="center", va="bottom", fontsize=8, color="#c1121f"
        )
    if sd_target_cm is not None:
        ax2.axhline(
            sd_target_cm,
            ls=":",
            c="k",
            alpha=0.8,
            label=f"16x4 reference config ({sd_target_cm:.0f})",
        )
    ax2.set_ylabel(r"recovered energy (cm$^{-1}$)", color="#c1121f")
    ax2.tick_params(axis="y", labelcolor="#c1121f")

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="center right")
    ax.set_title(
        "64-qubit 16x4: more shots do not rescue recovery\n"
        "100k -> 5M (50x) yields 0 -> 8 valid samples; energy stays ~6900 cm$^{-1}$"
    )
    fig.tight_layout()
    fig.savefig(outpath, dpi=130)
    plt.close(fig)


def plot_glycine64_recovered_dist(dist16, dist8, outpath, k=15):
    """Side-by-side stem plots of the recovered ground-state |c|^2 over the
    top-k configurations, 16x4 (left) vs 8x8 (right). Each dist maps a short
    config-label -> probability. Log y so the small tail is visible."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, dist, title in (
        (axes[0], dist16, "16 modes x 4 modal (modes 12-27, E~6885 cm$^{-1}$)"),
        (axes[1], dist8, "8 modes x 8 modal (modes 20-27, E~2096 cm$^{-1}$)"),
    ):
        items = sorted(dist.items(), key=lambda kv: kv[1], reverse=True)[:k]
        labels = [it[0] for it in items]
        probs = [max(it[1], 1e-6) for it in items]
        xs = np.arange(len(labels))
        ax.stem(xs, probs, basefmt=" ")
        ax.set_yscale("log")
        ax.set_ylim(1e-4, 1.5)
        ax.set_xticks(xs)
        ax.set_xticklabels(
            labels, rotation=60, ha="right", fontsize=6, family="monospace"
        )
        ax.set_ylabel(r"recovered $|c|^2$ (log)")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle(
        "64-qubit glycine: recovered ground-state configuration weights", fontsize=12
    )
    fig.tight_layout()
    fig.savefig(outpath, dpi=130)
    plt.close(fig)


# Backward-compatible aliases.
plot_imucj_math = plot_vimucj_math
plot_gucj_math = plot_vgucj_math


# --------------------------------------------------------------------------- #
# CH2O: 6-mode sweep across {2,4,6} modals x {HO,modal} bases x 5 ansatze
# --------------------------------------------------------------------------- #
def plot_ch2o_gate_cost_grid(cells, ansatze, outpath, molecule_label="CH2O", n_modes=6):
    """3x2 grid (modal count x basis) of grouped L3 CZ-count bars across
    ansatze. ``cells`` maps (n_modals, basis) -> {ansatz: (l3_cz, l3_depth)}."""
    modal_counts = sorted({k[0] for k in cells})
    bases = ["modal", "ho"]
    fig, axes = plt.subplots(
        len(modal_counts),
        len(bases),
        figsize=(11, 3.3 * len(modal_counts)),
        sharey=False,
    )
    x = np.arange(len(ansatze))
    for i, nm in enumerate(modal_counts):
        for j, basis in enumerate(bases):
            ax = axes[i, j]
            d = cells.get((nm, basis), {})
            cz = [d.get(a, (0, 0))[0] for a in ansatze]
            colors = [ANSATZ_COLORS.get(a, "#888888") for a in ansatze]
            ax.bar(x, cz, color=colors)
            for k, c in enumerate(cz):
                ax.text(k, c, str(c), ha="center", va="bottom", fontsize=7)
            ax.set_xticks(x)
            ax.set_xticklabels(ansatze, rotation=25, ha="right", fontsize=8)
            ax.set_title(f"{nm * n_modes}q, {basis} basis", fontsize=10)
            if j == 0:
                ax.set_ylabel("L3 CZ count")
    fig.suptitle(
        f"{molecule_label}: transpiled circuit cost (Heron L3) across modal counts and bases",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(outpath, dpi=130)
    plt.close(fig)


def plot_ch2o_energy_grid(
    cells, ansatze, outpath, exact_cm=None, molecule_label="CH2O", n_modes=6
):
    """3x2 grid (modal count x basis) of recovered-energy bars across ansatze,
    with the reference-config energy and (optional) classical Exact value as
    horizontal lines. ``cells`` maps (n_modals, basis) ->
    {"ref_cm": float, "energies": {ansatz: recovered_cm}}."""
    modal_counts = sorted({k[0] for k in cells})
    bases = ["modal", "ho"]
    fig, axes = plt.subplots(
        len(modal_counts),
        len(bases),
        figsize=(11, 3.3 * len(modal_counts)),
        sharey=False,
    )
    x = np.arange(len(ansatze))
    for i, nm in enumerate(modal_counts):
        for j, basis in enumerate(bases):
            ax = axes[i, j]
            d = cells.get((nm, basis))
            if d is None:
                ax.axis("off")
                continue
            energies = [d["energies"].get(a) for a in ansatze]
            colors = [ANSATZ_COLORS.get(a, "#888888") for a in ansatze]
            xs, ys, cs, labs = [], [], [], []
            for k, (a, e) in enumerate(zip(ansatze, energies)):
                if e is not None:
                    xs.append(k)
                    ys.append(e)
                    cs.append(colors[k])
                    labs.append(a)
            # Zoom the y-axis to the data's actual spread: all values here sit
            # within a few 10s of cm^-1 of each other, so a from-zero axis
            # crushes every meaningful difference into a flat wall of bars.
            all_vals = (
                list(ys) + [d["ref_cm"]] + ([exact_cm] if exact_cm is not None else [])
            )
            lo, hi = min(all_vals), max(all_vals)
            pad = max((hi - lo) * 0.35, 2.0)
            ax.bar(xs, ys, color=cs)
            for k, e in zip(xs, ys):
                ax.text(k, e, f"{e:.1f}", ha="center", va="bottom", fontsize=7)
            ax.axhline(
                d["ref_cm"],
                ls=":",
                c="k",
                alpha=0.8,
                label=f"reference config ({d['ref_cm']:.1f})",
            )
            if exact_cm is not None:
                ax.axhline(
                    exact_cm,
                    ls="--",
                    c="#c1121f",
                    alpha=0.8,
                    label=f"Exact ({exact_cm:.1f})",
                )
            ax.set_ylim(lo - pad, hi + pad)
            ax.set_xticks(x)
            ax.set_xticklabels(ansatze, rotation=25, ha="right", fontsize=8)
            ax.set_title(f"{nm * n_modes}q, {basis} basis", fontsize=10)
            ax.legend(fontsize=6, loc="upper right")
            if j == 0:
                ax.set_ylabel(r"recovered energy (cm$^{-1}$)")
    fig.suptitle(
        f"{molecule_label}: recovered energy vs reference / exact across modal counts and bases",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(outpath, dpi=130)
    plt.close(fig)


def plot_ch2o_quantum_vs_uniform(cells, outpath, molecule_label="CH2O", n_modes=6):
    """Grouped bar: best quantum-recovered energy vs uniform-random-recovered
    energy, per (modal count, basis). ``cells`` maps (n_modals, basis) ->
    {"quantum_cm": float, "uniform_cm": float, "ref_cm": float}."""
    keys = sorted(cells.keys())
    labels = [f"{nm * n_modes}q\n{basis}" for nm, basis in keys]
    x = np.arange(len(keys))
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.bar(
        x - 0.2,
        [cells[k]["quantum_cm"] for k in keys],
        0.4,
        label="best quantum-recovered",
        color="#c1121f",
    )
    ax.bar(
        x + 0.2,
        [cells[k]["uniform_cm"] for k in keys],
        0.4,
        label="uniform-random-recovered",
        color="#adb5bd",
    )
    ax.plot(
        x,
        [cells[k]["ref_cm"] for k in keys],
        "k_",
        markersize=20,
        label="reference config",
        zorder=5,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(r"recovered energy (cm$^{-1}$)")
    ax.set_yscale("log")
    ax.set_title(
        f"{molecule_label}: quantum sampling vs uniform-random control\n"
        "(same recovery loop, only the sampler differs)"
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outpath, dpi=130)
    plt.close(fig)


def plot_ch2o_beforeafter(before, after, outpath, title="", k=15):
    """Single before/after stem-plot pair (log y) for one CH2O cell. ``before``
    and ``after`` map bitstring/label -> weight."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, dist, sub in (
        (axes[0], before, "before (raw post-selected)"),
        (axes[1], after, "after (recovered)"),
    ):
        items = sorted(dist.items(), key=lambda kv: kv[1], reverse=True)[:k]
        labels = [it[0] for it in items]
        probs = [max(it[1], 1e-6) for it in items]
        xs = np.arange(len(labels))
        ax.stem(xs, probs, basefmt=" ")
        ax.set_yscale("log")
        ax.set_ylim(1e-4, 1.5)
        ax.set_xticks(xs)
        ax.set_xticklabels(
            labels, rotation=60, ha="right", fontsize=6, family="monospace"
        )
        ax.set_ylabel(r"$|c|^2$ (log)")
        ax.set_title(sub, fontsize=10)
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(outpath, dpi=130)
    plt.close(fig)


def plot_mode_marginals(before_occ, after_occ, mode_labels, outpath, title=""):
    """Grid of per-mode modal-occupation bar charts, before vs after SQD
    recovery. ``before_occ``/``after_occ`` are (n_modes, n_modals) marginal-
    probability arrays (e.g. from occupancies_from_raw_counts /
    occupancies_from_ground_state). ``mode_labels`` is a length-n_modes
    sequence of subplot titles."""
    n_modes, n_modals = before_occ.shape
    ncols = int(np.ceil(np.sqrt(n_modes)))
    nrows = int(np.ceil(n_modes / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.2 * ncols, 2.8 * nrows), squeeze=False
    )
    x = np.arange(n_modals)
    for m in range(n_modes):
        ax = axes[m // ncols][m % ncols]
        ax.bar(x - 0.2, before_occ[m], 0.4, color="#adb5bd", label="before")
        ax.bar(x + 0.2, after_occ[m], 0.4, color="#2a6f97", label="after")
        ax.set_xticks(x)
        ax.set_xticklabels([str(i) for i in x], fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_title(mode_labels[m], fontsize=9)
        ax.set_xlabel("modal index", fontsize=8)
        if m % ncols == 0:
            ax.set_ylabel("P(occupied)", fontsize=8)
        if m == 0:
            ax.legend(fontsize=7)
    for m in range(n_modes, nrows * ncols):
        axes[m // ncols][m % ncols].axis("off")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(outpath, dpi=130)
    plt.close(fig)
