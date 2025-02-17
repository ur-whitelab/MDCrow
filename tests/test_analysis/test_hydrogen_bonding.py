import mdtraj as md
import numpy as np
import pytest

from mdcrow.tools.base_tools.analysis_tools.hydrogen_bonding_tools import (
    HydrogenBondTool,
)


@pytest.fixture
def hydrogen_bond_tool(get_registry):
    path_registry = get_registry("raw", True)
    return HydrogenBondTool(path_registry)


@pytest.fixture
def dummy_traj():
    topology = md.Topology()
    chain = topology.add_chain()
    residue = topology.add_residue("ALA", chain)
    atom1 = topology.add_atom("N", element=md.element.nitrogen, residue=residue)
    atom2 = topology.add_atom("H", element=md.element.hydrogen, residue=residue)
    atom3 = topology.add_atom("O", element=md.element.oxygen, residue=residue)
    topology.add_bond(atom1, atom2)
    topology.add_bond(atom1, atom3)

    n_atoms = topology.n_atoms
    n_frames = 3
    coordinates = np.zeros((n_frames, n_atoms, 3))

    coordinates[0, :, :] = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
    coordinates[1, :, :] = [[0, 0, 0], [1.1, 0, 0], [0, 1.1, 0]]
    coordinates[2, :, :] = [[0, 0, 0], [1.2, 0, 0], [0, 1.2, 0]]

    traj = md.Trajectory(coordinates, topology)
    return traj


def test_compute_hbonds_traj(hydrogen_bond_tool, dummy_traj):
    hbond_counts = hydrogen_bond_tool.compute_hbonds_traj(dummy_traj)
    assert hbond_counts == [0, 0, 0]


def test_plot_hbonds_over_time(hydrogen_bond_tool, dummy_traj):
    hbond_counts = hydrogen_bond_tool.compute_hbonds_traj(dummy_traj)
    result = hydrogen_bond_tool.plot_hbonds_over_time(hbond_counts, dummy_traj, "dummy")
    assert "plot saved to" in result
    assert ".png" in result


def test_write_hbond_counts_to_file(hydrogen_bond_tool, dummy_traj):
    hbond_counts = hydrogen_bond_tool.compute_hbonds_traj(dummy_traj)
    result = hydrogen_bond_tool.write_hbond_counts_to_file(hbond_counts, "dummy")
    assert "Data saved to" in result
    assert ".csv" in result
