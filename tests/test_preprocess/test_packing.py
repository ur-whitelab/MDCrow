import os

import pytest

from mdcrow.tools.base_tools.preprocess_tools.packing import (
    Molecule,
    PackmolBox,
    PackMolTool,
)


@pytest.fixture
def packmolbox(get_registry):
    return PackmolBox(path_registry=get_registry("raw", False))


@pytest.fixture
def packmoltool(get_registry):
    return PackMolTool(path_registry=get_registry("raw", False))


@pytest.fixture
def dummy_molecule():
    return Molecule(
        filename="water", file_id=100, number_of_molecules=2, instructions=None
    )


@pytest.fixture
def packmol_valid_input():
    return {
        "pdbfiles_id": ["3pqr_test"],
        "small_molecules": ["water"],
        "number_of_molecules": [1, 2],
        "instructions": [
            ["fixed 0. 0. 0. 0. 0. 0. centerofmass"],
            ["inside box 0. 0. 0. 90. 90. 90."],
        ],
    }


def test_packmol_add_molecule(packmolbox, dummy_molecule):
    initial_length = len(packmolbox.molecules)
    packmolbox.add_molecule(dummy_molecule)
    assert len(packmolbox.molecules) == initial_length + 1


def test_packmol_generate_input_header(packmolbox):
    packmolbox.generate_input_header()
    # assert file packmol.inp exists
    assert os.path.isfile("packmol.inp")
    os.remove("packmol.inp")


def test_packmol_generate_input(packmolbox, dummy_molecule):
    packmolbox.add_molecule(dummy_molecule)
    output = packmolbox.generate_input()
    assert "structure water" in output
    assert "number 2" in output
    assert "end structure" in output


def test_packmol_validate_input_missing_info(packmoltool, packmol_valid_input):
    example_input = packmol_valid_input
    example_input["pdbfiles_id"] = []
    input_valid = packmoltool.validate_input(example_input)
    assert "error" in input_valid.keys()
    assert (
        "The length of number_of_molecules AND "
        "instructions must be equal to the number "
        "of species in the system. You have 1 from 0 "
        "pdbfiles and 1 small molecules" in input_valid["error"]
    )

    example_input["pdbfiles_id"] = ["nonsense"]
    input_valid = packmoltool.validate_input(example_input)
    assert "error" in input_valid.keys()
    assert (
        input_valid["error"]
        == "nonsense is not a valid pdbfile_id in the path_registry"
    )

    example_input["pdbfiles_id"] = ["3pqr_test"]
    example_input["small_molecules"] = ["water", "urea"]
    input_valid = packmoltool.validate_input(example_input)
    assert "error" in input_valid.keys()
    assert (
        "The length of number_of_molecules AND "
        "instructions must be equal to the number "
        "of species in the system. You have 3 from 1 "
        "pdbfiles and 2 small molecules" in input_valid["error"]
    )

    packmoltool.path_registry.map_path("3pqr_test", "3pqr.cif", "cif_test_file")
    packmoltool = PackMolTool(packmoltool.path_registry)
    example_input["pdbfiles_id"] = ["3pqr_test"]
    example_input["small_molecules"] = ["nonsense"]
    input_valid = packmoltool.validate_input(example_input)
    assert "error" in input_valid.keys()
    assert "nonsense could not be converted to a pdb file" in input_valid["error"]


def test_pacmol_validate_input_instruction_fail(packmoltool, packmol_valid_input):
    example_input = packmol_valid_input
    example_input["small_molecules"] = ["water"]
    example_input["instructions"] = [
        ["fail 0. 0. 0. 0. 0. 0. centerofmass"],
        ["inside box 0. 0. 0. 90. 90. 90."],
    ]
    input_valid = packmoltool.validate_input(example_input)
    assert "error" in input_valid.keys()
    assert "The first word of each instruction must be one of" in input_valid["error"]

    example_input["instructions"] = [["center"], [["inside box 0. 0. 0. 90. 90. 90."]]]
    input_valid = packmoltool.validate_input(example_input)
    assert "error" in input_valid.keys()
    assert (
        "The instruction 'center' must be accompanied by more instructions"
        in input_valid["error"]
    )

    example_input = packmol_valid_input
    example_input["instructions"] = [
        example_input["instructions"],
        example_input["instructions"],
    ]
    assert len(example_input["instructions"]) == 2
    input_valid = packmoltool.validate_input(example_input)
    assert "error" in input_valid.keys()
    assert "Each instruction must be a single string" in input_valid["error"]


def test_packmol_validate_input_valid(get_registry):
    registry = get_registry("raw", False)
    registry.map_path("3pqr_test", "3pqr.cif", "cif_test_file")
    registry.map_path("water", "water.pdb", "fake_water_test_file")
    packmoltool = PackMolTool(registry)
    example_input = {
        "pdbfiles_id": ["3pqr_test"],
        "small_molecules": ["water"],
        "number_of_molecules": [1, 2],
        "instructions": [
            ["fixed 0. 0. 0. 0. 0. 0. centerofmass"],
            ["inside box 0. 0. 0. 90. 90. 90."],
        ],
    }
    input_valid = packmoltool.validate_input(example_input)
    assert input_valid == example_input

    example_input["small_molecules"] = ["water", "urea"]
    registry.map_path("urea", "urea.pdb", "fake_urea_test_file")
    example_input["instructions"] = [
        example_input["instructions"][0],
        example_input["instructions"][0],
        example_input["instructions"][0],
    ]
    example_input["number_of_molecules"] = [1, 2, 2]
    input_valid = packmoltool.validate_input(example_input)
    assert input_valid == example_input
