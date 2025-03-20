from unittest.mock import patch

import mdtraj as md
import numpy as np
import pytest

from mdcrow.ldp_env.analysis_tools.rgy import compute_radius_of_gyration
from mdcrow.ldp_env.state import MDCrowState


@pytest.fixture
def state(get_registry):
    """Fixture for MDCrowState"""
    reg = get_registry("raw", True)
    return MDCrowState(path_registry=reg, tools=[])


@pytest.fixture
def mock_traj():
    """Fixture for a dummy trajectory with 5 residues and 3 frames"""
    top = md.Topology()
    chain = top.add_chain()
    for i in range(5):
        res = top.add_residue(f"RES{i}", chain)
        top.add_atom(f"CA{i}", md.element.carbon, res)

    traj = md.Trajectory(np.random.rand(3, 5, 3), top)
    return traj


def test_compute_rgy_valid(state, mock_traj):
    """Test `compute_radius_of_gyration()` with valid inputs"""
    with patch(
        "mdcrow.ldp_env.analysis_tools.rgy.load_single_traj", return_value=mock_traj
    ):
        message, _, _ = compute_radius_of_gyration(
            state,
            traj_file="mock_traj",
            top_file="mock_top",
        )
        assert "Succeeded." in message
        assert "Radii of gyration saved to " in message
        assert "Average radius of gyration: " in message
        assert "Plot saved as: " in message
        assert ".png" in message


def test_compute_rgy_invalid_topology(state):
    """Test `compute_radius_of_gyration()` with an invalid topology"""
    message, _, _ = compute_radius_of_gyration(
        state,
        traj_file="mock_traj",
        top_file="invalid_topology",
    )
    assert "Error loading traj" in message


def test_compute_rgy_per_frame(state, mock_traj):
    """Test `compute_radius_of_gyration()` per frame computation"""
    with patch(
        "mdcrow.ldp_env.analysis_tools.rgy.load_single_traj", return_value=mock_traj
    ):
        message, _, _ = compute_radius_of_gyration(
            state,
            traj_file="mock_traj",
            top_file="mock_top",
        )
        assert "Radii of gyration saved to " in message


def test_compute_rgy_average(state, mock_traj):
    """Test `compute_radius_of_gyration()` average computation"""
    with patch(
        "mdcrow.ldp_env.analysis_tools.rgy.load_single_traj", return_value=mock_traj
    ):
        message, _, _ = compute_radius_of_gyration(
            state,
            traj_file="mock_traj",
            top_file="mock_top",
        )
        assert "Average radius of gyration: " in message


def test_compute_rgy_plot(state, mock_traj):
    """Test `compute_radius_of_gyration()` plot generation"""
    with patch(
        "mdcrow.ldp_env.analysis_tools.rgy.load_single_traj", return_value=mock_traj
    ):
        message, _, _ = compute_radius_of_gyration(
            state,
            traj_file="mock_traj",
            top_file="mock_top",
        )
        assert "Plot saved as: " in message
        assert ".png" in message


def test_compute_rgy_full_analysis(state, mock_traj):
    """Test `compute_radius_of_gyration()` full computation with all outputs"""
    with patch(
        "mdcrow.ldp_env.analysis_tools.rgy.load_single_traj", return_value=mock_traj
    ):
        message, _, _ = compute_radius_of_gyration(
            state,
            traj_file="mock_traj",
            top_file="mock_top",
        )
        assert "Radii of gyration saved to " in message
        assert "Average radius of gyration: " in message
        assert "Plot saved as: " in message
        assert ".png" in message
