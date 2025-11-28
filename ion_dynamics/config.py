"""Configuration dataclasses for the 3D ion-transport simulator.

The defaults are chosen to reproduce the ferri/ferrocyanide experiments
reported by Utterback et al. (Operando Label-Free Optical Imaging of
Solution-Phase Ion Transport and Electrochemistry, 2023) and the values used
throughout their supporting information."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import jax.numpy as jnp


Vec3 = Tuple[float, float, float]


@dataclass
class SpeciesConfig:
    """Defines one ionic species."""

    name: str
    charge: float
    diffusion_cm2_s: float
    concentration_mM: float
    optical_coeff: float = 0.0
    adsorption_capacity_m2: float = 0.0  # Reserved for FeB/CrB particle studies

    def diffusion(self) -> float:
        return float(self.diffusion_cm2_s)

    def concentration(self) -> float:
        return float(self.concentration_mM) * 1.0e-6


@dataclass
class ButlerVolmerConfig:
    exchange_current_density: float = 17.0e-3  # A/cm^2
    alpha_a: float = 0.5
    alpha_c: float = 0.5
    formal_potential: float = -0.37  # V vs Ag/AgCl
    temperature_K: float = 298.0

    def rate_constant(self, red_conc: float) -> float:
        F = 96485.0
        return self.exchange_current_density / (F * red_conc)


@dataclass
class SimulationGridConfig:
    physical_size_um: Vec3 = (20.0, 20.0, 15.0)  # x, y, z (µm)
    shape: Tuple[int, int, int] = (64, 64, 96)


@dataclass
class TimeConfig:
    dt: float = 1.0e-6
    total_time: float = 0.1
    recording_period: float = 1.0e-3


@dataclass
class ObservationPlane:
    orientation: str = "xz"  # 'xy', 'xz', 'yz'
    position_um: float = 0.0
    thickness_um: float = 1.0


@dataclass
class OutputConfig:
    planes: Sequence[ObservationPlane] = field(
        default_factory=lambda: (
            ObservationPlane("xz", position_um=0.0, thickness_um=0.5),
            ObservationPlane("xy", position_um=10.0, thickness_um=0.5),
        )
    )
    optical_decay_um: float = 1.0
    save_3d_every: int = 10
    compute_volume_average: bool = True


@dataclass
class ElectrodeConfig:
    name: str
    role: str  # 'working', 'reference', 'counter'
    potential_scale: float = 1.0
    shape_factory: Optional[Callable[[Tuple[int, int, int]], jnp.ndarray]] = None


@dataclass
class SimulationConfig:
    species: Sequence[SpeciesConfig]
    grid: SimulationGridConfig = SimulationGridConfig()
    time: TimeConfig = TimeConfig()
    outputs: OutputConfig = OutputConfig()
    electrodes: Sequence[ElectrodeConfig] = field(default_factory=list)
    kinetics: ButlerVolmerConfig = ButlerVolmerConfig()
    supporting_electrolyte_mM: float = 100.0
    viscosity_Pa_s: float = 1.0e-3

    def baseline_species(self) -> Dict[str, SpeciesConfig]:
        return {sp.name: sp for sp in self.species}


def default_species() -> List[SpeciesConfig]:
    return [
        SpeciesConfig("[Fe(CN)6]4-", charge=-4, diffusion_cm2_s=8.96e-6, concentration_mM=30),
        SpeciesConfig("[Fe(CN)6]3-", charge=-3, diffusion_cm2_s=7.39e-6, concentration_mM=30),
        SpeciesConfig("K+", charge=1, diffusion_cm2_s=2.05e-5, concentration_mM=410),
        SpeciesConfig("SO4--", charge=-2, diffusion_cm2_s=1.07e-5, concentration_mM=100),
    ]


def default_config() -> SimulationConfig:
    grid = SimulationGridConfig()
    return SimulationConfig(
        species=default_species(),
        grid=grid,
        electrodes=[
            ElectrodeConfig(name="ITO", role="working"),
            ElectrodeConfig(name="Ag/AgCl", role="reference"),
            ElectrodeConfig(name="Pt mesh", role="counter"),
        ],
    )
