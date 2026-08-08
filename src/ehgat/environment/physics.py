"""SA-AGV physical constants and per-leg time/energy kinematics.

Grounded in Homayouni & Fontes (2022) Section 4 and Fontes & Homayouni (2022) Section 5:

    speed factor alpha:  v = alpha * v_nominal,   t = distance / v
    energy per leg:      E = power(kW) * travel_time(s)

Three discrete speed levels (V) are considered. Empty and loaded legs have distinct
nominal speeds and power draws; the +/-20% levels and their empirically-determined
power values are taken verbatim from the papers.

    Level    alpha   empty            loaded
    Lower    0.8     4.8 m/s, 7.8 kW  2.4 m/s, 11.7 kW
    Nominal  1.0     6.0 m/s, 10  kW  3.0 m/s, 15.0 kW
    Higher   1.2     7.2 m/s, 13.2 kW 3.6 m/s, 19.8 kW
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

__all__ = ["SPEED_TABLE", "SpeedLevel", "SpeedSpec", "leg_energy", "travel_time"]


class SpeedLevel(IntEnum):
    """Discrete SA-AGV speed levels (ordered low -> high, matching the BRKGA decoder)."""

    LOWER = 0
    NOMINAL = 1
    HIGHER = 2


@dataclass(frozen=True, slots=True)
class SpeedSpec:
    """Kinematic + power specification for one speed level."""

    alpha: float
    empty_speed: float  # m/s
    loaded_speed: float  # m/s
    empty_power: float  # kW (energy per second travelling empty)
    loaded_power: float  # kW (energy per second travelling loaded)


# Nominal references (Homayouni & Fontes 2022, Sec. 4).
_NOMINAL_EMPTY_SPEED = 6.0
_NOMINAL_LOADED_SPEED = 3.0

SPEED_TABLE: dict[SpeedLevel, SpeedSpec] = {
    SpeedLevel.LOWER: SpeedSpec(
        alpha=0.8, empty_speed=4.8, loaded_speed=2.4, empty_power=7.8, loaded_power=11.7
    ),
    SpeedLevel.NOMINAL: SpeedSpec(
        alpha=1.0, empty_speed=6.0, loaded_speed=3.0, empty_power=10.0, loaded_power=15.0
    ),
    SpeedLevel.HIGHER: SpeedSpec(
        alpha=1.2, empty_speed=7.2, loaded_speed=3.6, empty_power=13.2, loaded_power=19.8
    ),
}


def _validate_speed_table() -> None:
    """Fail loudly if the constants drift from the papers' kinematic relations.

    Enforces v = alpha * v_nominal for both empty and loaded references. Power
    values are empirical (not closed-form) and are therefore only range-checked.
    """
    for level, spec in SPEED_TABLE.items():
        expected_empty = spec.alpha * _NOMINAL_EMPTY_SPEED
        expected_loaded = spec.alpha * _NOMINAL_LOADED_SPEED
        if abs(spec.empty_speed - expected_empty) > 1e-9:
            raise ValueError(
                f"{level.name}: empty_speed {spec.empty_speed} != alpha*v0 {expected_empty}"
            )
        if abs(spec.loaded_speed - expected_loaded) > 1e-9:
            raise ValueError(
                f"{level.name}: loaded_speed {spec.loaded_speed} != alpha*v0 {expected_loaded}"
            )
        if not (0.0 < spec.empty_power < spec.loaded_power):
            raise ValueError(
                f"{level.name}: expected 0 < empty_power < loaded_power, got "
                f"{spec.empty_power} / {spec.loaded_power}"
            )


_validate_speed_table()


def travel_time(distance_m: float, level: SpeedLevel, *, loaded: bool) -> float:
    """Travel time (s) for a leg of distance_m metres at level.

    loaded selects the loaded vs empty speed for the leg.
    """
    if distance_m < 0.0:
        raise ValueError(f"distance must be non-negative, got {distance_m}")
    spec = SPEED_TABLE[level]
    speed = spec.loaded_speed if loaded else spec.empty_speed
    return distance_m / speed


def leg_energy(distance_m: float, level: SpeedLevel, *, loaded: bool) -> float:
    """Energy (kJ) consumed over a leg: power(kW) * travel_time(s)."""
    spec = SPEED_TABLE[level]
    power = spec.loaded_power if loaded else spec.empty_power
    return power * travel_time(distance_m, level, loaded=loaded)
