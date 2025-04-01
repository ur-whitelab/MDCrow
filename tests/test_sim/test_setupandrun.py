import json
import os

import pytest

from mdcrow.ldp_env.simulation_tools.set_up_and_run import (
    OpenMMSimulation,
    check_system_params,
)


@pytest.fixture(scope="module")
def raw():
    return "raw"


@pytest.fixture(scope="module")
def clean():
    return "clean"


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


def test_check_system_params(string_input, raw, clean):
    """Test the check_system_params method of the SetUpandRunFunction class."""

    # registry = get_registry(raw, False)
    # tool = SetUpandRunFunction(path_registry=registry)
    final_values_1 = check_system_params(json.loads(string_input(raw)))
    assert final_values_1.get("error") is None
    final_values_2 = check_system_params(json.loads(string_input(clean)))
    assert final_values_2.get("error") is None


def test_openmmsimulation_init(get_registry, string_input, raw, clean):
    """Test the OpenMMSimulation class initialization."""
    # assert an openmmexception is raised

    registry = get_registry(clean, True)
    tool_input = json.loads(string_input(clean))
    inputs = check_system_params(tool_input)
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
