import asyncio

import pytest

from mdcrow.ldp_env.analysis_tools.ppi_tools import ppi_distance
from mdcrow.ldp_env.analysis_tools.rmsd_tools import compute_rmsd, compute_rmsf, lprmsd
from mdcrow.ldp_env.state import MDCrowState
from mdcrow.ldp_env.utils import load_traj_with_ref

# PDB file with two chains
pdb_string = """
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 20.00           N
ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00 20.00           C
ATOM      3  C   ALA A   1       1.458   1.527   0.000  1.00 20.00           C
ATOM      4  O   ALA A   1       0.000   1.527   0.000  1.00 20.00           O
ATOM      5  CB  ALA A   1       1.458  -0.500   1.500  1.00 20.00           C
ATOM      6  N   GLY B   2      -1.458   0.000   0.000  1.00 20.00           N
ATOM      7  CA  GLY B   2      -2.916   0.000   0.000  1.00 20.00           C
ATOM      8  C   GLY B   2      -2.916   1.527   0.000  1.00 20.00           C
ATOM      9  O   GLY B   2      -1.458   1.527   0.000  1.00 20.00           O
ATOM     10  N   GLY B   3      -4.374   1.527   0.000  1.00 20.00           N
TER
END
"""


@pytest.fixture
def state(get_registry):
    reg = get_registry("raw", True)
    return MDCrowState(path_registry=reg, tools=[])


@pytest.fixture
def pdb_path(state):
    file_path = f"{state.path_registry.ckpt_dir}/twochains.pdb"
    with open(file_path, "w") as file:
        file.write(pdb_string)
    return file_path


@pytest.fixture
def get_trajs(state):
    traj, ref = load_traj_with_ref(
        state.path_registry, "top_sim0_butane_123456", "rec0_butane_123456"
    )
    return traj, ref


def test_ppi_distance(pdb_path):
    avg_dist = ppi_distance(pdb_path, "protein")
    assert avg_dist > 0, "Expected a positive average distance"


def test_ppi_distance_no_binding_residues(pdb_path):
    with pytest.raises(
        ValueError, match="No matching residues found for the binding site."
    ):
        ppi_distance(pdb_path, "residue 10000")


def test_ppi_distance_one_chain(state):
    file_path = state.path_registry.get_mapped_path("ALA_123456")
    with pytest.raises(
        ValueError, match="Only one chain found. Cannot compute PPI distance."
    ):
        ppi_distance(file_path, "protein")


def test_rmsd(state, get_trajs):
    message, _, _ = asyncio.run(
        compute_rmsd(
            state, "top_sim0_butane_123456", "rec0_butane_123456", mol_name="butane"
        )
    )
    assert "RMSD calculated and saved" in message


def test_rmsd_single_value(state):

    message, _, _ = asyncio.run(
        compute_rmsd(state, "top_sim0_butane_123456", mol_name="butane")
    )
    assert "RMSD calculated." in message


def test_rmsf(state):
    message, _, _ = asyncio.run(
        compute_rmsf(
            state,
            "top_sim0_butane_123456",
            "rec0_butane_123456",
            mol_name="butane",
            select="all",
        )
    )
    assert "RMSF calculated and saved" in message


def test_lprmsd(state, get_trajs):
    traj, ref_traj = get_trajs
    message = lprmsd(
        state.path_registry, traj, ref_traj, mol_name="butane", select="all"
    )
    assert "LP-RMSD calculated and saved" in message


def test_lprmsd_invalid_select(state, get_trajs):
    traj, ref_traj = get_trajs
    with pytest.raises(ValueError, match="No atoms found for selection 'protein'."):
        lprmsd(state.path_registry, traj, ref_traj, mol_name="butane", select="protein")
