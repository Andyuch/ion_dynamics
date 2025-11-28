"""Main 3D JAX simulation engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import jax
import jax.numpy as jnp
from tqdm import tqdm

from .config import ObservationPlane, SimulationConfig, SpeciesConfig
from .electrodes import mask_from_callable, semi_infinite_lower_plane
from .geometry import SimulationGrid, laplacian
from .observables import ObservationManager, compute_optical_signal, plane_average, volume_average


FARADAY = 96485.0
R_GAS = 8.314


def gradient(field: jnp.ndarray, spacing: Tuple[float, float, float]) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    grads = jnp.gradient(field, *spacing, edge_order=2)
    return tuple(grads)


def divergence(components: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray], spacing: Tuple[float, float, float]) -> jnp.ndarray:
    accum = jnp.zeros_like(components[0])
    for axis, (comp, h) in enumerate(zip(components, spacing)):
        deriv = jnp.gradient(comp, h, axis=axis, edge_order=2)
        accum = accum + deriv
    return accum


@dataclass
class SimulationResult:
    times: List[float] = field(default_factory=list)
    optical: List[float] = field(default_factory=list)
    current_density: List[float] = field(default_factory=list)
    plane_data: Dict[str, List[jnp.ndarray]] = field(default_factory=dict)
    volume_stats: List[Dict[str, float]] = field(default_factory=list)


class IonTransportSimulator:
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.grid = SimulationGrid(config.grid)
        self.spacing = self.grid.spacing_cm
        self.species = config.species
        self.n_species = len(self.species)
        self.diffusion = jnp.array([sp.diffusion() for sp in self.species])
        self.charges = jnp.array([sp.charge for sp in self.species])
        self.concentrations = self._initialize_concentrations()
        self.observer = ObservationManager(self.grid, config.outputs)
        self.plane_masks = self.observer.plane_masks()
        self.electrode_masks = self._build_electrodes()
        self.potential_bases = self._build_potential_bases()
        self.f_term = FARADAY / (R_GAS * config.kinetics.temperature_K)
        red_conc = self.species[0].concentration()
        self.k0 = config.kinetics.rate_constant(red_conc)
        self._compiled_step = jax.jit(self._step_impl)
        self.adsorption_layer = None

    def _initialize_concentrations(self) -> jnp.ndarray:
        grid_shape = (self.n_species,) + self.grid.shape
        conc = jnp.zeros(grid_shape)
        for idx, sp in enumerate(self.species):
            conc = conc.at[idx].set(sp.concentration())
        return conc

    def _build_electrodes(self) -> Dict[str, jnp.ndarray]:
        masks = {}
        shape = self.grid.shape
        for electrode in self.config.electrodes:
            if electrode.shape_factory is None and electrode.role == "working":
                mask = semi_infinite_lower_plane(shape)
            elif electrode.shape_factory is None and electrode.role == "reference":
                mask = jnp.zeros(shape).at[:, :, -1].set(1.0)
            else:
                mask = mask_from_callable(shape, electrode.shape_factory or (lambda s: semi_infinite_lower_plane(s)))
            masks[electrode.role] = mask
        if "reference" not in masks:
            masks["reference"] = jnp.zeros(shape).at[:, :, -1].set(1.0)
        return masks

    def _build_potential_bases(self) -> Dict[str, jnp.ndarray]:
        bases = {}
        reference = self.electrode_masks.get("reference", jnp.zeros(self.grid.shape))

        def relax(phi, fixed_mask, fixed_value, iters):
            boundary = jnp.clip(fixed_mask + reference, 0.0, 1.0)
            dirichlet = fixed_mask * fixed_value

            def body(_, val):
                padded = [
                    jnp.pad(val, ((1, 1), (0, 0), (0, 0)), mode="edge"),
                    jnp.pad(val, ((0, 0), (1, 1), (0, 0)), mode="edge"),
                    jnp.pad(val, ((0, 0), (0, 0), (1, 1)), mode="edge"),
                ]
                new_val = (
                    padded[0][2:, :, :] + padded[0][:-2, :, :] +
                    padded[1][:, 2:, :] + padded[1][:, :-2, :] +
                    padded[2][:, :, 2:] + padded[2][:, :, :-2]
                ) / 6.0
                return jnp.where(boundary > 0, dirichlet, new_val)

            return jax.lax.fori_loop(0, iters, body, phi)

        for role, mask in self.electrode_masks.items():
            if role == "reference":
                continue
            phi0 = jnp.zeros(self.grid.shape)
            bases[role] = relax(phi0, mask, 1.0, 800)
        bases["reference"] = self.electrode_masks["reference"]
        return bases

    def potential_field(self, potentials: Dict[str, float]) -> jnp.ndarray:
        phi = jnp.zeros(self.grid.shape)
        for role, base in self.potential_bases.items():
            phi = phi + potentials.get(role, 0.0) * base
        return phi

    def _step_impl(
        self,
        concentrations: jnp.ndarray,
        phi_field: jnp.ndarray,
        working_mask: jnp.ndarray,
        eta: float,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        updated = concentrations
        grad_phi = gradient(phi_field, tuple(self.spacing))
        spacing = tuple(self.spacing)
        for idx in range(self.n_species):
            lap = laplacian(updated[idx], spacing)
            migr = divergence(
                tuple(-self.diffusion[idx] * self.charges[idx] * updated[idx] * self.f_term * g for g in grad_phi),
                spacing,
            )
            updated = updated.at[idx].set(
                updated[idx] + self.config.time.dt * (self.diffusion[idx] * lap + migr)
            )

        exp_a = jnp.exp(self.config.kinetics.alpha_a * self.f_term * eta)
        exp_c = jnp.exp(-self.config.kinetics.alpha_c * self.f_term * eta)
        surf_red = updated[0][:, :, 0]
        surf_ox = updated[1][:, :, 0]
        mask2d = working_mask[:, :, 0]
        reaction_flux = self.k0 * (surf_red * exp_a - surf_ox * exp_c) * mask2d
        dz = self.spacing[2]
        delta = self.config.time.dt * reaction_flux / dz
        updated = updated.at[0, :, :, 0].add(-delta)
        updated = updated.at[1, :, :, 0].add(delta)
        updated = jnp.maximum(updated, 0.0)
        return updated, reaction_flux

    def run(self, voltage_program, with_progress: bool = True) -> SimulationResult:
        dt = self.config.time.dt
        steps = int(self.config.time.total_time / dt)
        record_interval = max(1, int(self.config.time.recording_period / dt))
        result = SimulationResult(plane_data={key: [] for key in self.plane_masks})
        iterator = range(steps)
        if with_progress:
            iterator = tqdm(iterator, desc="Ion transport")
        conc = self.concentrations
        for step in iterator:
            t = step * dt
            potentials = voltage_program.potentials(t)
            phi = self.potential_field(potentials)
            eta_val = jnp.array(potentials.get("working", 0.0) - self.config.kinetics.formal_potential)
            conc, reaction_flux = self._compiled_step(conc, phi, self.electrode_masks["working"], eta_val)
            if step % record_interval == 0:
                result.times.append(t)
                result.optical.append(
                    compute_optical_signal(conc, self.species, self.grid, self.config.outputs)
                )
                result.current_density.append(
                    float(FARADAY * jnp.mean(reaction_flux))
                )
                for key, mask in self.plane_masks.items():
                    orientation = key.split("_")[0]
                    result.plane_data[key].append(
                        plane_average(conc[0] * 1e6, mask, orientation)
                    )
                result.volume_stats.append(volume_average(conc, self.species))
        self.concentrations = conc
        return result

    def set_observation_planes(self, planes: List[ObservationPlane]) -> None:
        self.config.outputs.planes = planes
        self.observer = ObservationManager(self.grid, self.config.outputs)
        self.plane_masks = self.observer.plane_masks()

    def reset(self) -> None:
        self.concentrations = self._initialize_concentrations()

    def register_adsorption_sites(self, mask: jnp.ndarray, capacity_mM: float, rate: float) -> None:
        """Placeholder hook for FeB/CrB particle adsorption models."""
        self.adsorption_layer = {
            "mask": mask,
            "capacity": capacity_mM,
            "rate": rate,
        }
