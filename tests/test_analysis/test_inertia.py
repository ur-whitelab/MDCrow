from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from mdcrow.tools.base_tools.analysis_tools.inertia import MOIFunctions, MomentOfInertia


@pytest.fixture
def moi_functions(get_registry):
    registry = get_registry("raw", True)
    top_fileid = "top_sim0_butane_123456"
    traj_fileid = "rec0_butane_123456"
    return MOIFunctions(registry, top_fileid, traj_fileid)


def test_moi_tool_init(get_registry):
    registry = get_registry("raw", False)
    tool = MomentOfInertia(path_registry=registry)
    assert tool.name == "MomentOfInertia"
    assert tool.path_registry == registry


def test_calculate_moment_of_inertia(moi_functions):
    msg = moi_functions.calculate_moment_of_inertia()
    assert "Average Moment of Inertia" in msg
    assert "saved with file ID" in msg
    assert "MOI_sim0_butane" in msg

    moi_functions.mol_name = "butane"
    msg = moi_functions.calculate_moment_of_inertia()
    assert "MOI_butane" in msg


def test_plot_moi_one_frame(moi_functions):
    mocked_traj = MagicMock()
    mocked_traj.n_frames = 1  # Set the number of frames to 1
    moi_functions.traj = mocked_traj

    # Simulate a single frame of inertia tensor data
    moi_functions.moments_of_inertia = np.array([[1.0, 2.0, 3.0]])
    result = moi_functions.plot_moi()
    assert "Only one frame in trajectory, no plot generated." in result


@patch("mdcrow.tools.base_tools.analysis_tools.inertia.plt.savefig")
@patch("mdcrow.tools.base_tools.analysis_tools.inertia.plt.close")
def test_plot_moi_multiple_frames(mock_close, mock_savefig, moi_functions):
    # Simulate multiple frames of inertia tensor data
    moi_functions.moments_of_inertia = np.array([[1.0, 2.0, 3.0], [1.1, 2.1, 3.1]])
    moi_functions.avg_moi = np.mean(moi_functions.moments_of_inertia)
    moi_functions.min_moi = np.min(moi_functions.moments_of_inertia)

    result = moi_functions.plot_moi()
    assert "Plot of moments of inertia over time saved" in result
    mock_savefig.assert_called_once()
    mock_close.mock_close.call_count >= 1
