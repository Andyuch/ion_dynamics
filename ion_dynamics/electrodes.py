"""Electrode utilities and shape factories."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple

import jax.numpy as jnp


MaskFactory = Callable[[Tuple[int, int, int]], jnp.ndarray]


def semi_infinite_lower_plane(shape: Tuple[int, int, int]) -> jnp.ndarray:
    mask = jnp.zeros(shape)
    return mask.at[:, :, 0].set(1.0)


def rectangular_patch(
    shape: Tuple[int, int, int],
    x_um: Tuple[float, float],
    y_um: Tuple[float, float],
    grid_size_um: Tuple[float, float, float],
) -> jnp.ndarray:
    nx, ny, nz = shape
    x = jnp.linspace(0.0, grid_size_um[0], nx, endpoint=False)
    y = jnp.linspace(0.0, grid_size_um[1], ny, endpoint=False)
    xv, yv = jnp.meshgrid(x, y, indexing="ij")
    mask2d = jnp.logical_and(
        jnp.logical_and(xv >= x_um[0], xv <= x_um[1]),
        jnp.logical_and(yv >= y_um[0], yv <= y_um[1]),
    )
    mask = jnp.zeros(shape)
    return mask.at[:, :, 0].set(mask2d.astype(jnp.float32))


def circular_patch(
    shape: Tuple[int, int, int],
    center_um: Tuple[float, float],
    radius_um: float,
    grid_size_um: Tuple[float, float, float],
) -> jnp.ndarray:
    nx, ny, _ = shape
    x = jnp.linspace(0.0, grid_size_um[0], nx, endpoint=False)
    y = jnp.linspace(0.0, grid_size_um[1], ny, endpoint=False)
    xv, yv = jnp.meshgrid(x, y, indexing="ij")
    mask2d = (xv - center_um[0]) ** 2 + (yv - center_um[1]) ** 2 <= radius_um ** 2
    mask = jnp.zeros(shape)
    return mask.at[:, :, 0].set(mask2d.astype(jnp.float32))


def mask_from_callable(shape: Tuple[int, int, int], fn: MaskFactory) -> jnp.ndarray:
    mask = fn(shape)
    if mask.shape != shape:
        raise ValueError("Electrode mask has incompatible shape.")
    return mask.astype(jnp.float32)
