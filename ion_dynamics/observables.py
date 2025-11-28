"""Observable helpers for saving and plotting results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import jax.numpy as jnp

from .config import ObservationPlane, OutputConfig, SpeciesConfig
from .geometry import SimulationGrid


AXIS_MAP = {"xy": 2, "xz": 1, "yz": 0}


@dataclass
class ObservationManager:
    grid: SimulationGrid
    config: OutputConfig

    def plane_masks(self) -> Dict[str, jnp.ndarray]:
        masks = {}
        for idx, plane in enumerate(self.config.planes):
            masks[f"{plane.orientation}_{idx}"] = self.grid.plane_indices(plane)
        return masks


def plane_average(field: jnp.ndarray, mask: jnp.ndarray, orientation: str) -> jnp.ndarray:
    axis = AXIS_MAP[orientation]
    weighted = field * mask
    denom = jnp.maximum(jnp.sum(mask, axis=axis), 1e-12)
    return jnp.sum(weighted, axis=axis) / denom


def compute_optical_signal(
    concentrations: jnp.ndarray,
    species: Sequence[SpeciesConfig],
    grid: SimulationGrid,
    outputs: OutputConfig,
) -> float:
    z_axis = jnp.linspace(0.0, grid.config.physical_size_um[2], grid.shape[2], endpoint=False)
    weights = jnp.exp(-z_axis / outputs.optical_decay_um)
    weights = weights / jnp.sum(weights)
    optical = jnp.zeros(grid.shape)
    for idx, sp in enumerate(species):
        optical = optical + concentrations[idx] * sp.optical_coeff
    depth_projection = jnp.tensordot(optical, weights, axes=([2], [0]))
    return float(jnp.mean(depth_projection))


def current_density_from_flux(
    reaction_flux: jnp.ndarray,
    species: Sequence[SpeciesConfig],
    faraday: float = 96485.0,
) -> float:
    charge_transfer = sum(sp.charge for sp in species[:2])  # red/ox pair
    return float(faraday * jnp.mean(reaction_flux) * abs(charge_transfer))


def volume_average(concentrations: jnp.ndarray, species: Sequence[SpeciesConfig]) -> Dict[str, float]:
    averages = {}
    for idx, sp in enumerate(species):
        averages[sp.name] = float(jnp.mean(concentrations[idx]) * 1e6)
    return averages
