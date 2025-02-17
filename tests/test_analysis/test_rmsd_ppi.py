import pytest

from mdcrow.tools.base_tools.analysis_tools.ppi_tools import ppi_distance
from mdcrow.tools.base_tools.analysis_tools.rmsd_tools import lprmsd, rmsd, rmsf
from mdcrow.utils import load_traj_with_ref

# pdb with two chains
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
def pdb_path(get_registry):
    reg = get_registry("raw", True)
    file_path = f"{reg.ckpt_dir}/twochains.pdb"
    with open(file_path, "w") as file:
        file.write(pdb_string)
    return file_path


@pytest.fixture
def get_trajs(get_registry):
    reg = get_registry("raw", True)
    traj, ref = load_traj_with_ref(reg, "top_sim0_butane_123456", "rec0_butane_123456")
    return traj, ref


def test_ppi_distance(pdb_path):
    avg_dist = ppi_distance(pdb_path, "protein")
    assert avg_dist > 0, "Expected a positive average distance"


def test_ppi_distance_no_binding_residues(pdb_path):
    with pytest.raises(
        ValueError, match="No matching residues found for the binding site."
    ):
        ppi_distance(pdb_path, "residue 10000")


def test_ppi_distance_one_chain(get_registry):
    reg = get_registry("raw", True)
    file_path = reg.get_mapped_path("ALA_123456")
    with pytest.raises(
        ValueError, match="Only one chain found. Cannot compute PPI distance."
    ):
        ppi_distance(file_path, "protein")


def test_rmsd(get_registry, get_trajs):
    reg = get_registry("raw", False)
    traj, ref_traj = get_trajs
    result = rmsd(reg, traj, ref_traj, mol_name="butane")
    assert "RMSD calculated and saved" in result


def test_rmsd_single_value(get_registry):
    reg = get_registry("raw", True)
    traj, ref_traj = load_traj_with_ref(
        reg, "top_sim0_butane_123456", ignore_warnings=True
    )
    result = rmsd(reg, traj, ref_traj, mol_name="butane")
    assert "RMSD calculated." in result


def test_rmsf(
    get_registry,
):
    reg = get_registry("raw", True, dynamic=True, include_hydrogens=True)
    traj, ref = load_traj_with_ref(reg, "top_sim0_butane_123456", "rec0_butane_123456")
    result = rmsf(reg, traj, ref, mol_name="butane", select="all")
    assert "RMSF calculated and saved" in result


def test_lprmsd(get_registry, get_trajs):
    reg = get_registry("raw", False)
    traj, ref_traj = get_trajs
    result = lprmsd(reg, traj, ref_traj, mol_name="butane", select="all")
    assert "LP-RMSD calculated and saved" in result


def test_lprmsd_invalid_select(get_registry, get_trajs):
    reg = get_registry("raw", False)
    traj, ref_traj = get_trajs
    with pytest.raises(ValueError, match="No atoms found for selection 'protein'."):
        lprmsd(reg, traj, ref_traj, mol_name="butane", select="protein")
