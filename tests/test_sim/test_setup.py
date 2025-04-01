import pytest
from openmm import unit
from openmm.app import PME, HBonds

from mdcrow.ldp_env.simulation_tools.set_up_and_run import (
    _parse_cutoff,
    _process_parameters,
    parse_friction,
    parse_pressure,
    parse_temperature,
    parse_timestep,
)


@pytest.mark.parametrize(
    "input_cutoff, expected_result",
    [
        (unit.Quantity(1.0, unit.nanometers), unit.Quantity(1.0, unit.nanometers)),
        (3.0, unit.Quantity(3.0, unit.nanometers)),
        ("2angstroms", unit.Quantity(2.0, unit.angstroms)),
    ],
)
def test_parse_cutoff(input_cutoff, expected_result):
    result = _parse_cutoff(input_cutoff)
    assert expected_result == result


def test_parse_cutoff_unknown_unit():
    with pytest.raises(ValueError) as e:
        _parse_cutoff("2pc")
    assert "Unknown unit" in str(e.value)


def test_parse_temperature():
    result = parse_temperature("300k")
    result2 = parse_temperature("300kelvin")
    expected_result = unit.Quantity(300, unit.kelvin)
    assert expected_result == result[0] == result2[0]


@pytest.mark.parametrize(
    "input_friction, expected_friction_result",
    [
        ("1/ps", unit.Quantity(1, 1 / unit.picoseconds)),
        ("1/picosecond", unit.Quantity(1, 1 / unit.picosecond)),
        ("1/picoseconds", unit.Quantity(1, 1 / unit.picosecond)),
        ("1picosecond^-1", unit.Quantity(1, 1 / unit.picosecond)),
        ("1picoseconds^-1", unit.Quantity(1, 1 / unit.picoseconds)),
        ("1/ps^-1", unit.Quantity(1, 1 / unit.picoseconds)),
        ("1ps^-1", unit.Quantity(1, 1 / unit.picoseconds)),
        ("1*ps^-1", unit.Quantity(1, 1 / unit.picoseconds)),
    ],
)
def test_parse_friction(input_friction, expected_friction_result):
    result = parse_friction(input_friction)
    assert (
        expected_friction_result == result[0]
    ), f"Expected {expected_friction_result} for {input_friction}, got {result[0]}"


@pytest.mark.parametrize(
    "input_time, expected_time_unit",
    [
        ("1ps", unit.picoseconds),
        ("1picosecond", unit.picoseconds),
        ("1picoseconds", unit.picoseconds),
        ("1fs", unit.femtoseconds),
        ("1femtosecond", unit.femtoseconds),
        ("1femtoseconds", unit.femtoseconds),
        ("1ns", unit.nanoseconds),
        ("1nanosecond", unit.nanoseconds),
        ("1nanoseconds", unit.nanoseconds),
    ],
)
def test_parse_time(input_time, expected_time_unit):
    result = parse_timestep(input_time)
    expected_result = unit.Quantity(1, expected_time_unit)
    assert expected_result == result[0]


@pytest.mark.parametrize(
    "input_pressure, expected_pressure_unit",
    [
        ("1bar", unit.bar),
        ("1atm", unit.atmospheres),
        ("1atmosphere", unit.atmospheres),
        ("1pascal", unit.pascals),
        ("1pascals", unit.pascals),
        ("1Pa", unit.pascals),
        ("1poundforce/inch^2", unit.psi),
        ("1psi", unit.psi),
    ],
)
def test_parse_pressure(input_pressure, expected_pressure_unit):
    result = parse_pressure(input_pressure)
    expected_result = unit.Quantity(1, expected_pressure_unit)
    # assert expected_result == result[0]
    if expected_result != result[0]:
        raise AssertionError(
            f"Expected {expected_result} for {input_pressure}, got {result[0]}"
        )


def test_process_parameters():
    parameters = {
        "nonbondedMethod": "PME",
        "constraints": "HBonds",
        "rigidWater": True,
    }
    result = _process_parameters(parameters)
    expected_result = {
        "nonbondedMethod": PME,
        "constraints": HBonds,
        "rigidWater": True,
    }
    for key in expected_result:
        assert result[0][key] == expected_result[key]
