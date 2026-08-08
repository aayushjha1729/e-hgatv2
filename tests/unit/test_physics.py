"""Unit tests for SA-AGV physics constants and kinematics."""

from __future__ import annotations

import math

import pytest

from ehgat.environment.physics import (
    SPEED_TABLE,
    SpeedLevel,
    leg_energy,
    travel_time,
)


def test_speed_table_matches_paper_constants() -> None:
    lower = SPEED_TABLE[SpeedLevel.LOWER]
    nominal = SPEED_TABLE[SpeedLevel.NOMINAL]
    higher = SPEED_TABLE[SpeedLevel.HIGHER]

    assert (nominal.empty_speed, nominal.loaded_speed) == (6.0, 3.0)
    assert (nominal.empty_power, nominal.loaded_power) == (10.0, 15.0)
    assert (lower.empty_speed, lower.loaded_speed) == (4.8, 2.4)
    assert (lower.empty_power, lower.loaded_power) == (7.8, 11.7)
    assert (higher.empty_speed, higher.loaded_speed) == (7.2, 3.6)
    assert (higher.empty_power, higher.loaded_power) == (13.2, 19.8)


@pytest.mark.parametrize("level", list(SpeedLevel))
def test_alpha_speed_relation(level: SpeedLevel) -> None:
    spec = SPEED_TABLE[level]
    assert math.isclose(spec.empty_speed, spec.alpha * 6.0, rel_tol=0, abs_tol=1e-9)
    assert math.isclose(spec.loaded_speed, spec.alpha * 3.0, rel_tol=0, abs_tol=1e-9)


def test_travel_time_and_energy_nominal_empty() -> None:
    # 300 m empty at nominal 6 m/s = 50 s; energy = 10 kW * 50 s = 500.
    assert math.isclose(travel_time(300.0, SpeedLevel.NOMINAL, loaded=False), 50.0)
    assert math.isclose(leg_energy(300.0, SpeedLevel.NOMINAL, loaded=False), 500.0)


def test_travel_time_and_energy_nominal_loaded() -> None:
    # 300 m loaded at nominal 3 m/s = 100 s; energy = 15 kW * 100 s = 1500.
    assert math.isclose(travel_time(300.0, SpeedLevel.NOMINAL, loaded=True), 100.0)
    assert math.isclose(leg_energy(300.0, SpeedLevel.NOMINAL, loaded=True), 1500.0)


def test_zero_distance_is_zero_cost() -> None:
    assert travel_time(0.0, SpeedLevel.HIGHER, loaded=True) == 0.0
    assert leg_energy(0.0, SpeedLevel.LOWER, loaded=False) == 0.0


def test_negative_distance_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        travel_time(-1.0, SpeedLevel.NOMINAL, loaded=False)
