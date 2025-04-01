import asyncio
import os
from unittest.mock import MagicMock, patch

import pytest

from mdcrow.ldp_env.preprocess_tools import (
    MolPDB,
    download_pdb_file,
    get_small_molecule_PDB,
    pack_molecules,
)
from mdcrow.ldp_env.state import MDCrowState


@pytest.fixture
def fibronectin():
    return "fibronectin pdb"


@pytest.fixture
def molpdb(get_registry):
    return MolPDB(get_registry("raw", False))


def test_getpdb(fibronectin, get_registry):
    state = MDCrowState([], path_registry=get_registry("raw", False))
    name, _, _ = asyncio.run(download_pdb_file(fibronectin, state))
    assert name.startswith("Succeeded.")


@pytest.mark.parametrize("small_molecule", [["water"], ["benzene"]])
def test_get_small_molecule(small_molecule, get_registry):
    state = MDCrowState([], path_registry=get_registry("raw", False))
    state.path_registry._remove_path_from_json(f"{small_molecule[0]}")

    asyncio.run(get_small_molecule_PDB(state, small_molecule[0]))

    here = os.path.exists(f"{state.path_registry.ckpt_pdb}/{small_molecule[0]}.pdb")
    os.path.exists(f"{state.path_registry.ckpt_pdb}/{small_molecule[0]}.pdb")
    assert here  # or maybe_here


def test_small_molecule_pdb(molpdb):
    # Test with a valid SMILES string
    valid_smiles = "C1=CC=CC=C1"  # Benzene
    expected_output = "PDB file for C1=CC=CC=C1 successfully created and saved to "
    valid_pdb = molpdb.small_molecule_pdb(valid_smiles)
    assert expected_output in valid_pdb and "benzene.pdb" in valid_pdb
    assert os.path.exists(f"{molpdb.path_registry.ckpt_pdb}/benzene.pdb")

    # test with invalid SMILES string and invalid molecule name
    invalid_smiles = "C1=CC=CC=C1X"
    invalid_name = "NotAMolecule"
    expected_output = (
        "There was an error getting pdb. Please input a single molecule name."
    )
    assert expected_output in molpdb.small_molecule_pdb(invalid_smiles)
    assert expected_output in molpdb.small_molecule_pdb(invalid_name)

    # test with valid molecule name
    valid_name = "water"
    expected_output = (
        "Succeeded. PDB file for water successfully " "created and saved to water.pdb."
    )

    assert molpdb.small_molecule_pdb(valid_name) == expected_output
    assert os.path.exists(f"{molpdb.path_registry.ckpt_pdb}/water.pdb")


def test_packmol_sm_download_called(get_registry):
    state = MDCrowState([], path_registry=get_registry("raw", False))
    state.path_registry.map_path(
        "1A3N_144150", f"{state.path_registry.ckpt_pdb}/1A3N_144150.pdb", "pdb"
    )
    with patch(
        "mdcrow.ldp_env.preprocess_tools.packing._get_sm_pdbs",
        new=MagicMock(),
    ) as mock_get_sm_pdbs:
        test_values = {
            "pdbfile_ids": ["1A3N_144150"],
            "small_molecules": ["water", "benzene"],
            "number_of_molecules": [1, 10, 10],
            "instructions": [
                ["inside box 0. 0. 0. 100. 100. 100."],
                ["inside box 0. 0. 0. 100. 100. 100."],
                ["inside box 0. 0. 0. 100. 100. 100."],
            ],
        }

        pack_molecules(state, **test_values)

        mock_get_sm_pdbs.assert_called_with(state, ["water", "benzene"])


cids = {
    "CO": 887,
    "CCO": 702,
    "O": 962,
    "CC(=O)C": 180,
    "C(=O)(N)N": 1176,
    "CS(=O)C": 679,
    "CN(C)C=O": 6228,
    "C(C(CO)O)O": 753,
}


def get_cid(smiles):
    return cids[smiles]


pairs = [
    ("CO", "MOH"),
    ("CCO", "EOH"),
    ("O", "HOH"),
    ("CC(=O)C", "ACN"),
    ("C(=O)(N)N", "URE"),
    ("CS(=O)C", "DMS"),
    ("CN(C)C=O", "DMF"),
    ("CCO", "EOH"),
    ("C(C(CO)O)O", "GOL"),
]


@pytest.mark.parametrize("smiles, codes", pairs)
def test_get_het_codes(molpdb, smiles, codes):
    cid = get_cid(smiles)  # to not test the get_cid function
    assert molpdb.get_hetcode_from_cid(cid) == codes
