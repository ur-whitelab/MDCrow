import mdtraj as md
import numpy as np
import pytest

from mdcrow.tools.base_tools.analysis_tools.secondary_structure import (
    ComputeAcylindricity,
    ComputeAsphericity,
    ComputeDSSP,
    ComputeGyrationTensor,
    ComputeRelativeShapeAntisotropy,
    SummarizeProteinStructure,
)


@pytest.fixture
def compute_dssp_simple(get_registry):
    registry = get_registry("raw", True)
    return ComputeDSSP(path_registry=registry, simplified=True)


@pytest.fixture
def compute_dssp(get_registry):
    registry = get_registry("raw", True)
    return ComputeDSSP(path_registry=registry, simplified=False)


def test_compute_dssp(loaded_cif_traj, compute_dssp_simple, compute_dssp):
    dssp_simple = compute_dssp_simple._compute_dssp(loaded_cif_traj)
    assert dssp_simple.shape[1] == 374
    assert np.all(
        dssp_simple[0][:10]
        == np.array(["C", "C", "C", "E", "E", "E", "C", "C", "E", "E"])
    )

    dssp = compute_dssp._compute_dssp(loaded_cif_traj)
    assert dssp.shape[1] == 374
    assert np.all(dssp[0][:10] == [" ", " ", " ", "E", "E", "E", "T", "T", "E", "E"])


def test_get_frame(compute_dssp):
    # random dummy traj with 3 frames
    xyz = np.random.rand(10, 10, 3)
    topology = md.Topology()
    chain = topology.add_chain()
    residue = topology.add_residue("ALA", chain)
    for _ in range(10):
        topology.add_atom("CA", md.element.carbon, residue)
    traj = md.Trajectory(xyz, topology)

    # first frame
    first_frame = compute_dssp._get_frame(traj, "first")
    assert first_frame.n_frames == 1
    assert np.array_equal(first_frame.xyz, traj.xyz[0].reshape(1, -1, 3))

    # last frame
    last_frame = compute_dssp._get_frame(traj, "last")
    assert last_frame.n_frames == 1
    assert np.array_equal(last_frame.xyz, traj.xyz[-1].reshape(1, -1, 3))

    # all frames
    all_frames = compute_dssp._get_frame(traj, "all")
    assert all_frames.n_frames == traj.n_frames
    assert np.array_equal(all_frames.xyz, traj.xyz)


def test_dssp_codes(compute_dssp_simple, compute_dssp):
    dssp_codes_simple = compute_dssp_simple._dssp_codes()
    assert dssp_codes_simple == ["H", "E", "C", "NA"]

    nl_simple = compute_dssp_simple._dssp_natural_language()
    assert nl_simple == {
        "H": "residues in helix",
        "E": "residues in strand",
        "C": "residues in coil",
        "NA": "residues not assigned, not a protein residue",
    }

    dssp_codes = compute_dssp._dssp_codes()
    assert dssp_codes == ["H", "B", "E", "G", "I", "T", "S", " ", "NA"]

    nl = compute_dssp._dssp_natural_language()
    assert nl == {
        "H": "residues in alpha helix",
        "B": "residues in beta bridge",
        "E": "residues in extended strand",
        "G": "residues in three helix",
        "I": "residues in five helix",
        "T": "residues in hydrogen bonded turn",
        "S": "residues in bend",
        " ": "residues in loop or irregular",
        "NA": "residues not assigned, not a protein residue",
    }


def test_convert_dssp_counts(compute_dssp_simple, compute_dssp):
    dssp_counts = {"H": 0, "E": 5, "C": 5}

    descriptive_counts = compute_dssp_simple._convert_dssp_counts(dssp_counts)
    assert descriptive_counts == {
        "residues in helix": 0,
        "residues in strand": 5,
        "residues in coil": 5,
    }

    dssp_counts = {"H": 0, "B": 0, "E": 5, "G": 0, "I": 0, "T": 0, "S": 0, " ": 5}

    descriptive_counts = compute_dssp._convert_dssp_counts(dssp_counts)

    assert descriptive_counts == {
        "residues in alpha helix": 0,
        "residues in beta bridge": 0,
        "residues in extended strand": 5,
        "residues in three helix": 0,
        "residues in five helix": 0,
        "residues in hydrogen bonded turn": 0,
        "residues in bend": 0,
        "residues in loop or irregular": 5,
    }


def test_summarize_dssp(compute_dssp_simple, compute_dssp):
    dssp_array = np.array([["C", "C", "C", "E", "E", "E", "C", "C", "E", "E"]])
    summary = compute_dssp_simple._summarize_dssp(dssp_array)
    assert summary == {
        "residues in helix": 0,
        "residues in strand": 5,
        "residues in coil": 5,
        "residues not assigned, not a protein residue": 0,
    }

    dssp_array = np.array([[" ", " ", " ", "E", "E", "E", "T", "T", "E", "E"]])
    summary = compute_dssp._summarize_dssp(dssp_array)
    assert summary == {
        "residues in alpha helix": 0,
        "residues in beta bridge": 0,
        "residues in extended strand": 5,
        "residues in three helix": 0,
        "residues in five helix": 0,
        "residues in hydrogen bonded turn": 2,
        "residues in bend": 0,
        "residues in loop or irregular": 3,
        "residues not assigned, not a protein residue": 0,
    }


def test_compute_gyration_tensor(get_registry, loaded_cif_traj):
    gyration_tensor = np.array(
        [
            [
                [3.45897484, 0.17571401, -0.08759158],
                [0.17571401, 0.944077, 0.17698189],
                [-0.08759158, 0.17698189, 0.73760228],
            ]
        ]
    )
    registry = get_registry("raw", True)
    gy_tensor = ComputeGyrationTensor(path_registry=registry)._compute_gyration_tensor(
        loaded_cif_traj
    )
    assert np.allclose(gy_tensor, gyration_tensor)


def test_compute_asphericity(get_registry, loaded_cif_traj):
    registry = get_registry("raw", True)
    asphericity = ComputeAsphericity(path_registry=registry)
    output = asphericity._compute_asphericity(loaded_cif_traj)
    assert np.allclose(output, np.array([2.63956945]))


def test_compute_acylindricity(get_registry, loaded_cif_traj):
    registry = get_registry("raw", True)
    acylindricity = ComputeAcylindricity(path_registry=registry)
    output = acylindricity._compute_acylindricity(loaded_cif_traj)
    assert np.allclose(output, np.array([0.41455165]))


def test_compute_relative_shape_antisotropy(get_registry, loaded_cif_traj):
    registry = get_registry("raw", True)
    compute_relative_shape_antisotropy = ComputeRelativeShapeAntisotropy(
        path_registry=registry
    )
    output = compute_relative_shape_antisotropy._compute_relative_shape_antisotropy(
        loaded_cif_traj
    )
    assert np.allclose(output, np.array([0.26852832]))


def test_get_protein_stats(get_registry):
    registry = get_registry("raw", True)
    get_stats = SummarizeProteinStructure(registry)

    n_atoms = 5
    n_frames = 2
    coordinates = np.random.random((n_frames, n_atoms, 3))

    topology = md.Topology()

    for _ in range(n_atoms):
        topology.add_atom(
            "C",
            md.element.carbon,
            topology.add_residue("methane", topology.add_chain()),
        )

    bonds = [(0, i) for i in range(1, n_atoms)]
    for bond in bonds:
        topology.add_bond(topology.atom(bond[0]), topology.atom(bond[1]))

    traj = md.Trajectory(coordinates, topology)
    assert get_stats.summarize_protein_structure(
        traj, ["atoms", "residues", "chains", "frames", "bonds"]
    ) == {"n_atoms": 5, "n_residues": 5, "n_chains": 5, "n_frames": 2, "n_bonds": 4}

    # without topology
    traj = md.Trajectory(coordinates, None)
    with pytest.raises(ValueError):
        get_stats.summarize_protein_structure(
            traj, ["atoms", "residues", "chains", "frames", "bonds"]
        )
