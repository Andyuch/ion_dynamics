"""Spatial grid helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import jax.numpy as jnp

from .config import ObservationPlane, SimulationGridConfig


CM_PER_UM = 1.0e-4


@dataclass
class SimulationGrid:
    config: SimulationGridConfig

    @property
    def shape(self) -> Tuple[int, int, int]:
        return self.config.shape

    @property
    def spacing_cm(self) -> Tuple[float, float, float]:
        size_cm = tuple(v * CM_PER_UM for v in self.config.physical_size_um)
        return (
            size_cm[0] / self.shape[0],
            size_cm[1] / self.shape[1],
            size_cm[2] / self.shape[2],
        )

    def coordinates_um(self) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        grid = tuple(jnp.linspace(0.0, size, num, endpoint=False) for size, num in zip(self.config.physical_size_um, self.shape))
        return jnp.meshgrid(*grid, indexing="ij")

    def plane_indices(self, plane: ObservationPlane) -> jnp.ndarray:
        x, y, z = self.coordinates_um()
        if plane.orientation == "xy":
            dist = jnp.abs(z - plane.position_um)
        elif plane.orientation == "xz":
            dist = jnp.abs(y - plane.position_um)
        else:
            dist = jnp.abs(x - plane.position_um)
        return jnp.where(dist <= plane.thickness_um / 2.0, 1.0, 0.0)


def zero_flux_pad(arr: jnp.ndarray, axis: int) -> jnp.ndarray:
    pad = [(0, 0)] * arr.ndim
    pad[axis] = (1, 1)
    return jnp.pad(arr, pad, mode="edge")


def laplacian(arr: jnp.ndarray, spacing: Tuple[float, float, float]) -> jnp.ndarray:
    dx, dy, dz = spacing
    padded = [zero_flux_pad(arr, axis=i) for i in range(3)]
    term_x = (padded[0][2:, :, :] - 2 * arr + padded[0][:-2, :, :]) / (dx ** 2)
    term_y = (padded[1][:, 2:, :] - 2 * arr + padded[1][:, :-2, :]) / (dy ** 2)
    term_z = (padded[2][:, :, 2:] - 2 * arr + padded[2][:, :, :-2]) / (dz ** 2)
    return term_x + term_y + term_z
