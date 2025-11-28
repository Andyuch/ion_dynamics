"""Voltage waveforms for the 3-electrode system."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Sequence

import jax.numpy as jnp


class VoltageProgram:
    """Base class returning potentials for each electrode."""

    def roles(self) -> Sequence[str]:
        return ("working", "reference", "counter")

    def potentials(self, t: float) -> Dict[str, float]:
        raise NotImplementedError

    def sample(self, times: Iterable[float]) -> Dict[str, jnp.ndarray]:
        arrays = {role: [] for role in self.roles()}
        for t in times:
            values = self.potentials(float(t))
            for role in arrays:
                arrays[role].append(values.get(role, 0.0))
        return {role: jnp.array(vals) for role, vals in arrays.items()}


@dataclass
class PotentialStep(VoltageProgram):
    initial: float
    step_to: float
    step_time: float
    counter_shift: float = 0.0

    def potentials(self, t: float) -> Dict[str, float]:
        value = self.initial if t < self.step_time else self.step_to
        return {
            "working": value,
            "reference": 0.0,
            "counter": value + self.counter_shift,
        }


@dataclass
class CyclicVoltammogram(VoltageProgram):
    start: float
    lower: float
    upper: float
    scan_rate: float
    cycles: int = 1
    counter_shift: float = 0.1

    def durations(self) -> float:
        return 2.0 * abs(self.upper - self.lower) / self.scan_rate

    def potentials(self, t: float) -> Dict[str, float]:
        cycle_time = self.durations()
        total = self.cycles * cycle_time
        t_mod = jnp.clip(t % total, 0.0, cycle_time)
        half = cycle_time / 2.0
        if t_mod <= half:
            value = self.lower + self.scan_rate * t_mod
        else:
            value = self.upper - self.scan_rate * (t_mod - half)
        return {
            "working": value,
            "reference": 0.0,
            "counter": value + self.counter_shift,
        }


@dataclass
class CustomWaveform(VoltageProgram):
    timeline: Sequence[float] = field(default_factory=list)
    working: Sequence[float] = field(default_factory=list)
    counter: Sequence[float] = field(default_factory=list)

    def potentials(self, t: float) -> Dict[str, float]:
        if not self.timeline:
            return {"working": 0.0, "reference": 0.0, "counter": 0.0}
        idx = jnp.clip(jnp.searchsorted(jnp.array(self.timeline), t) - 1, 0, len(self.timeline) - 1)
        return {
            "working": float(self.working[idx]),
            "reference": 0.0,
            "counter": float(self.counter[idx]) if self.counter else 0.0,
        }
