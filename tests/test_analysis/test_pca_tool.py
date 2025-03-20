from unittest.mock import patch

import mdtraj as md
import numpy as np
import pytest

from mdcrow.ldp_env.analysis_tools.pca_tools import PCA_analysis, perform_pca_analysis
from mdcrow.ldp_env.state import MDCrowState


@pytest.fixture
def state(get_registry):
    reg = get_registry("raw", True, dynamic=True, include_hydrogens=True)
    return MDCrowState(path_registry=reg, tools=[])


def test_pca_tool_bad_inputs(state):
    bad_input_files = {
        "trajectory_fileid": "rec0_butane_456456",
        "topology_fileid": "top_sim0_butane_456456",
        "pc_percentage": "Ninety Percent",
        "analysis": "all",
        "selection": "name CA",
    }

    message, _, _ = perform_pca_analysis(state, **bad_input_files)

    assert "Trajectory File ID not in path registry" in message
    assert "Topology File ID not in path registry" in message
    assert "pc_percentage value must be a float" in message


def test_pca_tool_good_inputs(state):
    good_inputs = {
        "trajectory_fileid": "rec0_butane_123456",
        "topology_fileid": "top_sim0_butane_123456",
        "pc_percentage": "95",
        "analysis": "all",
        "selection": "all",
    }

    with patch("matplotlib.pyplot.savefig"), patch("matplotlib.pyplot.close"), patch(
        "seaborn.PairGrid"
    ), patch("seaborn.PairGrid.map"):
        message, _, _ = perform_pca_analysis(state, **good_inputs)

    assert "Analyses done:" in message
    assert "Cosine Content of each PC: " in message


def test_cosine_content(state):
    traj = md.load(
        state.path_registry.get_mapped_path("rec0_butane_123456"),
        top=state.path_registry.get_mapped_path("top_sim0_butane_123456"),
    )

    pca_analysis = PCA_analysis(
        path_registry=state.path_registry,
        pc_percentage=95,
        traj=traj,
        sim_id="sim0_butane_123456",
        selection="all",
    )

    pca_space = np.random.rand(100, 2)

    assert pca_analysis._cosine_content(pca_space=pca_space, i=0) < 0.1
    assert pca_analysis._cosine_content(pca_space=pca_space, i=1) < 0.1


def test_sub_array_sum(state):
    traj = md.load(
        state.path_registry.get_mapped_path("rec0_butane_123456"),
        top=state.path_registry.get_mapped_path("top_sim0_butane_123456"),
    )

    pca_analysis = PCA_analysis(
        path_registry=state.path_registry,
        pc_percentage=95,
        traj=traj,
        sim_id="sim0_butane_123456",
        selection="all",
    )

    array_1 = [0.60, 0.25, 0.11, 0.04]
    array_2 = [0.30, 0.25, 0.25, 0.20]
    array_3 = [0.96, 0.02, 0.01, 0.01]
    array_4 = [0.01, 0.01, 0.01, 0.01, 0.01]

    assert pca_analysis._sub_array_sum_to_m(array_1, 0.95) == [0.60, 0.25, 0.11]
    assert pca_analysis._sub_array_sum_to_m(array_2, 0.95) == [0.30, 0.25, 0.25, 0.20]
    assert pca_analysis._sub_array_sum_to_m(array_3, 0.95) == [0.96]
    assert pca_analysis._sub_array_sum_to_m(array_4, 0.95) == array_4
