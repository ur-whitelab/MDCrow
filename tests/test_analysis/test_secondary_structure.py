import asyncio
from unittest.mock import patch

import mdtraj as md
import numpy as np
import pytest

from mdcrow.ldp_env.analysis_tools.secondary_structures import (
    _compute_dssp,
    computeDSSP,
    summarize_protein_structure,
)
from mdcrow.ldp_env.state import MDCrowState


@pytest.fixture
def state(get_registry):
    reg = get_registry("raw", True)
    return MDCrowState(path_registry=reg, tools=[])


@pytest.fixture
def mock_traj():
    # Create a mock trajectory with 3 frames and 5 residues
    top = md.Topology()
    chain = top.add_chain()
    for i in range(5):
        res = top.add_residue(f"RES{i}", chain)
        _ = top.add_atom(f"CA{i}", md.element.carbon, res)

    traj = md.Trajectory(np.random.rand(3, 5, 3), top)
    return traj


def test_dssp_tool_valid_input(state, mock_traj):
    with patch(
        "mdcrow.ldp_env.analysis_tools.secondary_structures.load_single_traj",
        return_value=mock_traj,
    ):
        message, _, _ = asyncio.run(
            computeDSSP(
                state,
                traj_file="mock_traj",
                top_file="mock_top",
                target_frames="all",
                simplified=True,
            )
        )
        assert "Succeeded." in message


def test_dssp_tool_invalid_topology(state):
    message, _, _ = asyncio.run(
        computeDSSP(
            state,
            traj_file="mock_traj",
            top_file="invalid_topology",
            target_frames="all",
            simplified=True,
        )
    )
    assert "Failed." in message


def test_dssp_tool_first_frame(state, mock_traj):
    with patch(
        "mdcrow.ldp_env.analysis_tools.secondary_structures.load_single_traj",
        return_value=mock_traj,
    ):
        message, _, _ = asyncio.run(
            computeDSSP(
                state,
                traj_file="mock_traj",
                top_file="mock_top",
                target_frames="first",
                simplified=True,
            )
        )
        assert "Succeeded." in message


def test_dssp_tool_last_frame(state, mock_traj):
    with patch(
        "mdcrow.ldp_env.analysis_tools.secondary_structures.load_single_traj",
        return_value=mock_traj,
    ):
        message, _, _ = asyncio.run(
            computeDSSP(
                state,
                traj_file="mock_traj",
                top_file="mock_top",
                target_frames="last",
                simplified=True,
            )
        )
        assert "Succeeded." in message


def test_dssp_tool_invalid_target_frames(state, mock_traj):
    with patch(
        "mdcrow.ldp_env.analysis_tools.secondary_structures.load_single_traj",
        return_value=mock_traj,
    ):
        message, _, _ = asyncio.run(
            computeDSSP(
                state,
                traj_file="mock_traj",
                top_file="mock_top",
                target_frames="invalid_frame",
                simplified=True,
            )
        )
        assert "Failed." in message


def test_dssp_tool_summary(state, mock_traj):
    with patch(
        "mdcrow.ldp_env.analysis_tools.secondary_structures.load_single_traj",
        return_value=mock_traj,
    ):
        summary = _compute_dssp(mock_traj, simplified=True)
        assert summary.shape == (3, 5)  # Expecting a (frames, residues) shape


def test_summarize_protein_structure_valid(state, mock_traj):
    with patch(
        "mdcrow.ldp_env.analysis_tools.secondary_structures.load_single_traj",
        return_value=mock_traj,
    ):
        message, _, _ = asyncio.run(
            summarize_protein_structure(
                state, traj_file="mock_traj", top_file="mock_top"
            )
        )
        assert "Succeeded." in message
        assert "n_atoms" in message
        assert "n_residues" in message
        assert "n_chains" in message
        assert "n_frames" in message
        assert "n_bonds" in message


def test_summarize_protein_structure_invalid(state):
    message, _, _ = asyncio.run(
        summarize_protein_structure(
            state, traj_file="mock_traj_invalid", top_file="mock_top"
        )
    )
    assert "Failed." in message


def test_summarize_protein_structure_specific_analyses(state, mock_traj):
    with patch(
        "mdcrow.ldp_env.analysis_tools.secondary_structures.load_single_traj",
        return_value=mock_traj,
    ):
        message, _, _ = asyncio.run(
            summarize_protein_structure(
                state,
                traj_file="mock_traj",
                top_file="mock_top",
                requested_analyses=["atoms", "frames"],
            )
        )
        assert "Succeeded." in message
        assert "n_atoms" in message
        assert "n_frames" in message
        assert "n_residues" not in message
        assert "n_chains" not in message
