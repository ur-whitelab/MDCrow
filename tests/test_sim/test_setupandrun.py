import json
import os

import pytest
from openmm import unit
from openmm.app import PME, HBonds

from mdcrow.tools.base_tools.simulation_tools.setup_and_run import (
    OpenMMSimulation,
    SetUpandRunFunction,
)


@pytest.fixture(scope="module")
def raw():
    return "raw"


@pytest.fixture(scope="module")
def clean():
    return "clean"


@pytest.fixture
def setupandrun(get_registry):
    return SetUpandRunFunction(get_registry("raw", False))


@pytest.fixture(scope="module")
def string_input():
    def create_input(raw_or_clean):
        if raw_or_clean == "raw":
            pdb_id = "ALA_123456"
        elif raw_or_clean == "clean":
            pdb_id = "ALA_654321"
        return """
        {{
            "pdb_id": "{pdb_id}",
            "forcefield_files": ["amber14-all.xml", "amber14/tip3pfb.xml"],
            "save": true,
            "system_params":{{
                "nonbondedMethod": "PME",
                "nonbondedCutoff": "1 * nanometers",
                "ewaldErrorTolerance": 0.0005,
                "constraints": "HBonds",
                "rigidWater": true,
                "constraintTolerance": 0.00001,
                "solvate": false
            }},
            "integrator_params":{{
                "integrator_type": "LangevinMiddle",
                "Temperature": "300 * kelvin",
                "Friction": "1 / picosecond",
                "Timestep": "2 * femtoseconds"
            }},
                "simulation_params": {{
                "Ensemble": "NVT",
                "Number of Steps": 5000,
                "record_interval_steps": 50,
                "record_params": ["step", "potentialEnergy", "temperature"]
            }}
        }}
        """.format(
            pdb_id=pdb_id
        ).strip()

    return create_input


def test_init_SetUpandRunFunction(get_registry):
    """Test the SetUpandRunFunction class initialization."""
    registry = get_registry("raw", False)
    tool = SetUpandRunFunction(path_registry=registry)
    assert tool.name == "SetUpandRunFunction"
    assert tool.path_registry == registry


def test_check_system_params(get_registry, string_input, raw, clean):
    """Test the check_system_params method of the SetUpandRunFunction class."""

    registry = get_registry(raw, False)
    tool = SetUpandRunFunction(path_registry=registry)
    final_values_1 = tool.check_system_params(json.loads(string_input(raw)))
    assert final_values_1.get("error") is None
    final_values_2 = tool.check_system_params(json.loads(string_input(clean)))
    assert final_values_2.get("error") is None


def test_openmmsimulation_init(get_registry, string_input, raw, clean):
    """Test the OpenMMSimulation class initialization."""
    # assert an openmmexception is raised

    registry = get_registry(clean, True)
    tool_input = json.loads(string_input(clean))
    inputs = SetUpandRunFunction(path_registry=registry).check_system_params(tool_input)
    registry.get_mapped_path(tool_input["pdb_id"])
    Simulation = OpenMMSimulation(
        input_params=inputs,
        path_registry=registry,
        save=tool_input["save"],
        sim_id="sim_654321",
        pdb_id=tool_input["pdb_id"],
    )
    assert Simulation.save == tool_input["save"]
    assert Simulation.sim_id == "sim_654321"
    assert Simulation.pdb_id == tool_input["pdb_id"]
    assert Simulation.path_registry == registry

    # remove files that start with LOG, TOP, and TRAJ
    for file in os.listdir("."):
        if file.startswith("LOG") or file.startswith("TOP") or file.startswith("TRAJ"):
            os.remove(f"{file}")


@pytest.mark.parametrize(
    "input_cutoff, expected_result",
    [
        (unit.Quantity(1.0, unit.nanometers), unit.Quantity(1.0, unit.nanometers)),
        (3.0, unit.Quantity(3.0, unit.nanometers)),
        ("2angstroms", unit.Quantity(2.0, unit.angstroms)),
    ],
)
def test_parse_cutoff(setupandrun, input_cutoff, expected_result):
    result = setupandrun._parse_cutoff(input_cutoff)
    assert expected_result == result


def test_parse_cutoff_unknown_unit(setupandrun):
    with pytest.raises(ValueError) as e:
        setupandrun._parse_cutoff("2pc")
        assert "Unknown unit" in str(e.value)


def test_parse_temperature(setupandrun):
    result = setupandrun.parse_temperature("300k")
    result2 = setupandrun.parse_temperature("300kelvin")
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
def test_parse_friction(setupandrun, input_friction, expected_friction_result):
    result = setupandrun.parse_friction(input_friction)
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
def test_parse_time(setupandrun, input_time, expected_time_unit):
    result = setupandrun.parse_timestep(input_time)
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
def test_parse_pressure(setupandrun, input_pressure, expected_pressure_unit):
    result = setupandrun.parse_pressure(input_pressure)
    expected_result = unit.Quantity(1, expected_pressure_unit)
    # assert expected_result == result[0]
    if expected_result != result[0]:
        raise AssertionError(
            f"Expected {expected_result} for {input_pressure}, got {result[0]}"
        )


def test_process_parameters(setupandrun):
    parameters = {
        "nonbondedMethod": "PME",
        "constraints": "HBonds",
        "rigidWater": True,
    }
    result = setupandrun._process_parameters(parameters)
    expected_result = {
        "nonbondedMethod": PME,
        "constraints": HBonds,
        "rigidWater": True,
    }
    for key in expected_result:
        assert result[0][key] == expected_result[key]
