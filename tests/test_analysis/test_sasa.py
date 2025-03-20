import warnings
from unittest.mock import patch

import numpy as np
import pytest

from mdcrow.ldp_env.analysis_tools.sasa import (
    SASAFunctions,
    compute_solvent_accessible_surface_area,
)
from mdcrow.ldp_env.state import MDCrowState


@pytest.fixture
def state(get_registry):
    reg = get_registry("raw", True)
    return MDCrowState(path_registry=reg, tools=[])


@pytest.fixture
def sasa_functions_with_files(state):
    return SASAFunctions(
        state.path_registry, "top_sim0_butane_123456", "rec0_butane_123456"
    )


def test_sasa_analysis_init_success(state):
    with patch.object(
        state.path_registry,
        "get_mapped_path",
        wraps=state.path_registry.get_mapped_path,
    ) as mocked_get_mapped_path:
        analysis = SASAFunctions(
            state.path_registry, "top_sim0_butane_123456", "rec0_butane_123456"
        )
        assert mocked_get_mapped_path.call_count == 2
        assert analysis.path_registry == state.path_registry
        assert analysis.molecule_name == "sim0_butane_123456"
        assert analysis.traj is not None


def test_sasa_analysis_init_success_no_traj(state):
    with patch.object(
        state.path_registry,
        "get_mapped_path",
        wraps=state.path_registry.get_mapped_path,
    ) as mocked_get_mapped_path:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            analysis = SASAFunctions(
                state.path_registry, "top_sim0_butane_123456", mol_name="butane"
            )
        mocked_get_mapped_path.assert_called_once()
        assert analysis.path_registry == state.path_registry
        assert analysis.molecule_name == "butane"


def test_sasa_tool_valid_input(state):
    message, _, _ = compute_solvent_accessible_surface_area(
        state,
        top_fileid="top_sim0_butane_123456",
        traj_fileid="rec0_butane_123456",
        molecule_name="butane",
    )
    assert "Succeeded." in message
    assert "SASA analysis completed." in message


def test_sasa_tool_missing_trajectory(state):
    message, _, _ = compute_solvent_accessible_surface_area(
        state,
        top_fileid="top_sim0_butane_123456",  # No trajectory file
        molecule_name="butane",
    )
    assert "Succeeded." in message
    assert "SASA values computed and saved" in message


def test_sasa_tool_invalid_topology(state):
    message, _, _ = compute_solvent_accessible_surface_area(
        state,
        top_fileid="invalid_topology_id",
        traj_fileid="rec0_butane_123456",
        molecule_name="butane",
    )
    assert "Failed." in message
    assert "Error" in message


def test_sasa_tool_single_frame(state):
    message, _, _ = compute_solvent_accessible_surface_area(
        state,
        top_fileid="top_sim0_butane_123456",
        # traj_fileid="single_frame_trajectory",
        molecule_name="butane",
    )
    assert "Succeeded." in message
    assert "Only one frame in trajectory. No SASA plot generated." in message


@patch("mdcrow.ldp_env.analysis_tools.sasa.np.savetxt")
def test_calculate_sasa(mock_savetxt, sasa_functions_with_files):
    analysis = sasa_functions_with_files
    result = analysis.calculate_sasa()
    assert "SASA values computed and saved" in result
    mock_savetxt.assert_called_once()
    assert analysis.residue_sasa is not None
    assert analysis.total_sasa is not None


@patch("mdcrow.ldp_env.analysis_tools.sasa.plt.savefig")
@patch("mdcrow.ldp_env.analysis_tools.sasa.plt.close")
def test_plot_sasa(mock_close, mock_savefig, sasa_functions_with_files):
    analysis = sasa_functions_with_files
    analysis.residue_sasa = np.array([[1, 2], [3, 4]])
    analysis.total_sasa = np.array([1, 2])
    with patch.object(SASAFunctions, "calculate_sasa") as mock_calc:
        result = analysis.plot_sasa()
        mock_calc.assert_not_called()
        mock_savefig.assert_called_once()
        assert "SASA analysis completed" in result
