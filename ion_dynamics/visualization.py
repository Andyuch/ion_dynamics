"""Visualization utilities for 2D and pseudo-3D outputs."""

from __future__ import annotations

from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np


def plot_plane_field(field: np.ndarray, orientation: str, name: str, time: float) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(field.T, origin="lower", aspect="auto", cmap="viridis")
    ax.set_title(f"{name} ({orientation}) @ {time*1e3:.1f} ms")
    fig.colorbar(im, ax=ax, label="Concentration (mM)")
    return fig


def plot_volume_isosurface(field: np.ndarray, iso_value: float, name: str) -> plt.Figure:
    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(111, projection="3d")
    filled = np.argwhere(field >= iso_value)
    scatter = ax.scatter(filled[:, 0], filled[:, 1], filled[:, 2], s=2, alpha=0.3, c=filled[:, 2])
    ax.set_title(f"Isosurface for {name} at {iso_value:.2f} mM")
    fig.colorbar(scatter, ax=ax, shrink=0.6)
    return fig
