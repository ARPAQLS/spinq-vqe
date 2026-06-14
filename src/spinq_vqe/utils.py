"""
utils.py
--------
Plotting helpers and Pauli string utilities for spinq-vqe.

Functions
---------
plot_kagome_graph        : Visualize the Kagome lattice with sublattice colors
plot_energy_convergence  : Plot VQE energy history for HEA vs MERA
plot_entanglement_profile: Plot S_vN vs bipartition size
plot_mutual_info_matrix  : Heatmap of sublattice mutual information
plot_gradient_variance   : Barren plateau diagnostic plot
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# ---------------------------------------------------------------------------
# Soft pastel palette
# ---------------------------------------------------------------------------

# Sublattice colors: muted lavender, warm peach, soft sage
SUBLATTICE_COLORS = {0: "#B8B8E8", 1: "#F5C9A0", 2: "#A8D8B0"}
SUBLATTICE_LABELS = {0: "A", 1: "B", 2: "C"}

# Ansatz trace colors: dusty blue and muted coral
ANSATZ_COLORS = {"hea": "#7EB8D4", "mera": "#E8A598"}

# Accent / reference line
REF_COLOR = "#B0B0B0"

# Heatmap colormap (soft warm ramp)
HEATMAP_CMAP = "YlOrBr"

# Plot style — light, clean, publication-ready
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.facecolor": "#FAFAFA",
        "figure.facecolor": "#FFFFFF",
        "axes.edgecolor": "#CCCCCC",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "axes.grid": True,
        "grid.color": "#EBEBEB",
        "grid.linewidth": 0.7,
        "xtick.color": "#666666",
        "ytick.color": "#666666",
        "axes.labelcolor": "#444444",
        "text.color": "#333333",
        "figure.dpi": 120,
    }
)


# ---------------------------------------------------------------------------
# Lattice visualization
# ---------------------------------------------------------------------------


def plot_kagome_graph(
    G: nx.Graph,
    title: str = "Kagome Lattice",
    figsize: tuple = (8, 4),
    save_path: str | None = None,
) -> plt.Figure:
    """
    Draw the Kagome lattice graph with sublattice coloring.

    Parameters
    ----------
    G : nx.Graph
        Kagome graph from :func:`spinq_vqe.kagome.kagome_graph`.
    title : str
    figsize : tuple
    save_path : str or None
        If given, save the figure to this path.

    Returns
    -------
    plt.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Layout: position sites along a horizontal strip
    n_sites = G.number_of_nodes()
    n_cells = n_sites // 3
    pos = {}
    for k in range(n_cells):
        pos[3 * k] = (3 * k, 0.0)
        pos[3 * k + 1] = (3 * k + 1.0, 0.0)
        pos[3 * k + 2] = (3 * k + 0.5, 0.866)  # equilateral triangle

    node_colors = [SUBLATTICE_COLORS[G.nodes[n]["sublattice"]] for n in G.nodes]
    node_labels = {
        n: f"{n}\n({SUBLATTICE_LABELS[G.nodes[n]['sublattice']]})" for n in G.nodes
    }

    nx.draw_networkx(
        G,
        pos=pos,
        ax=ax,
        node_color=node_colors,
        node_size=650,
        labels=node_labels,
        font_size=8,
        font_color="#444444",
        font_weight="semibold",
        edge_color="#D0D0D0",
        width=1.8,
    )

    # Legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(
            facecolor=SUBLATTICE_COLORS[i],
            edgecolor="#AAAAAA",
            label=f"Sublattice {SUBLATTICE_LABELS[i]}",
        )
        for i in range(3)
    ]
    ax.legend(
        handles=legend_elements,
        loc="upper right",
        framealpha=0.9,
        edgecolor="#DDDDDD",
        fontsize=10,
    )
    ax.set_title(title, fontsize=13, fontweight="semibold", color="#333333")
    ax.set_facecolor("#FAFAFA")
    ax.axis("off")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    return fig


# ---------------------------------------------------------------------------
# VQE convergence
# ---------------------------------------------------------------------------


def plot_energy_convergence(
    results: dict,
    ed_energy: float | None = None,
    title: str = "VQE Energy Convergence",
    figsize: tuple = (9, 4),
    save_path: str | None = None,
) -> plt.Figure:
    """
    Plot energy histories for HEA and MERA VQE runs.

    Parameters
    ----------
    results : dict
        {"hea": VQEResult, "mera": VQEResult} from :func:`spinq_vqe.vqe.compare_ansatze`.
    ed_energy : float or None
        Exact diagonalization energy (horizontal reference line).
    title, figsize, save_path : as usual.

    Returns
    -------
    plt.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    labels = {"hea": "HEA", "mera": "MERA (simplified)"}

    for name, result in results.items():
        ax.plot(
            result.energy_history,
            color=ANSATZ_COLORS[name],
            label=f"{labels[name]}  (final: {result.energy:.5f})",
            linewidth=2.0,
            alpha=0.9,
        )

    if ed_energy is not None:
        ax.axhline(
            ed_energy,
            color=REF_COLOR,
            linestyle="--",
            linewidth=1.4,
            label=f"ED exact: {ed_energy:.5f}",
        )

    ax.set_xlabel("Optimizer step", color="#555555")
    ax.set_ylabel("Energy (normalized)", color="#555555")
    ax.set_title(title, fontsize=13, fontweight="semibold", color="#333333")
    ax.legend(framealpha=0.9, edgecolor="#DDDDDD", fontsize=10)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    return fig


# ---------------------------------------------------------------------------
# Entanglement entropy profile
# ---------------------------------------------------------------------------


def plot_entanglement_profile(
    profile: dict,
    ansatz_label: str = "",
    figsize: tuple = (7, 4),
    save_path: str | None = None,
) -> plt.Figure:
    """
    Plot von Neumann entropy vs subsystem size.

    Parameters
    ----------
    profile : dict
        Output of :func:`spinq_vqe.entanglement.entanglement_profile`.
    ansatz_label : str
    figsize, save_path : as usual.
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(
        profile["subsystem_sizes"],
        profile["entropies"],
        "o-",
        color="#9BB8D4",  # soft dusty blue
        linewidth=2.0,
        markersize=7,
        markerfacecolor="#D4E6F1",
        markeredgecolor="#9BB8D4",
        markeredgewidth=1.2,
        label=ansatz_label or "VQE",
    )

    ax.set_xlabel("Subsystem size |A|", color="#555555")
    ax.set_ylabel("$S_{\\mathrm{vN}}(\\rho_A)$ [bits]", color="#555555")
    ax.set_title(
        "Entanglement Entropy Profile",
        fontsize=13,
        fontweight="semibold",
        color="#333333",
    )
    if ansatz_label:
        ax.legend(framealpha=0.9, edgecolor="#DDDDDD", fontsize=10)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    return fig


# ---------------------------------------------------------------------------
# Sublattice mutual information heatmap
# ---------------------------------------------------------------------------


def plot_mutual_info_matrix(
    matrix: np.ndarray,
    figsize: tuple = (5, 4),
    save_path: str | None = None,
) -> plt.Figure:
    """
    Heatmap of the 3×3 sublattice mutual information matrix.

    Parameters
    ----------
    matrix : np.ndarray, shape (3, 3)
        Output of :func:`spinq_vqe.entanglement.sublattice_mutual_info_matrix`.
    """
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(matrix, cmap=HEATMAP_CMAP, vmin=0, alpha=0.85)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("I(A:B) [bits]", color="#555555", fontsize=10)
    cbar.ax.yaxis.set_tick_params(color="#888888")

    sl_labels = ["A", "B", "C"]
    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 1, 2])
    ax.set_xticklabels(sl_labels, color="#555555")
    ax.set_yticklabels(sl_labels, color="#555555")

    vmax = matrix.max() if matrix.max() > 0 else 1.0
    for i in range(3):
        for j in range(3):
            ax.text(
                j,
                i,
                f"{matrix[i, j]:.3f}",
                ha="center",
                va="center",
                fontsize=10,
                color="#555555" if matrix[i, j] < vmax * 0.65 else "#F5F0EB",
            )

    ax.set_title(
        "Sublattice Mutual Information I(A:B)",
        fontsize=12,
        fontweight="semibold",
        color="#333333",
    )
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    return fig


# ---------------------------------------------------------------------------
# Barren plateau diagnostic
# ---------------------------------------------------------------------------


def plot_gradient_variance(
    results: dict,
    figsize: tuple = (9, 4),
    save_path: str | None = None,
) -> plt.Figure:
    """
    Plot gradient variance over training steps for HEA vs MERA.
    Vanishing gradient variance = barren plateau signature.
    """
    fig, ax = plt.subplots(figsize=figsize)
    labels = {"hea": "HEA", "mera": "MERA (simplified)"}

    for name, result in results.items():
        ax.semilogy(
            result.gradient_variance_history,
            color=ANSATZ_COLORS[name],
            label=labels[name],
            linewidth=2.0,
            alpha=0.9,
        )

    ax.set_xlabel("Optimizer step", color="#555555")
    ax.set_ylabel("Gradient variance (log scale)", color="#555555")
    ax.set_title(
        "Barren Plateau Diagnostic: Gradient Variance",
        fontsize=13,
        fontweight="semibold",
        color="#333333",
    )
    ax.legend(framealpha=0.9, edgecolor="#DDDDDD", fontsize=10)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    return fig
