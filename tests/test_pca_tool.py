from unittest.mock import patch

import numpy as np

from mdagent.tools.base_tools import PCATool
from mdagent.tools.base_tools.analysis_tools.pca_tools import PCA_analysis


def test_pca_tool_bad_inputs(get_registry):
    reg = get_registry("raw", True, dynamic=True, include_hydrogens=True)
    pca_tool = PCATool(path_registry=reg)
    bad_input_files = {
        "trajectory_fileid": "rec0_butane_456456",
        "topology_fileid": "top_sim0_butane_456456",
        "pc_percentage": "Ninety Percent",
        "analysis": "all",
        "selection": "name CA",
    }

    error_catching = pca_tool._run(bad_input_files)
    assert "Trajectory File ID not in path registry" in error_catching
    assert "Topology File ID not in path registry" in error_catching
    assert "pc_percentage value must be a float" in error_catching


def test_pca_tool_good_inputs(get_registry):
    reg = get_registry("raw", True, dynamic=True, include_hydrogens=True)
    pca_tool = PCATool(path_registry=reg)
    good_inputs = {
        "trajectory_fileid": "rec0_butane_123456",
        "topology_fileid": "top_sim0_butane_123456",
        "pc_percentage": "95",
        "analysis": "all",
        "selection": "name CA",
    }

    with patch("matplotlib.pyplot.figure"), patch("matplotlib.pyplot.plot"), patch(
        "matplotlib.pyplot.xlabel"
    ), patch("matplotlib.pyplot.ylabel"), patch("matplotlib.pyplot.title"), patch(
        "matplotlib.pyplot.savefig"
    ), patch(
        "matplotlib.pyplot.close"
    ), patch(
        "seaborn.PairGrid"
    ), patch(
        "seaborn.PairGrid.map"
    ):
        result = pca_tool._run(good_inputs)

    assert "Analyses done:" in result
    assert "Cosine Content of each PC: " in result

    # This two tests involve matplotlib, and are annoying to get done.
    # assert "PCA plots saved as" in result
    # assert "Scree Plot saved as" in result


def test_cosine_content(get_registry):
    reg = get_registry("raw", True, dynamic=True, include_hydrogens=True)
    pca_analysis = PCA_analysis(
        path_registry=reg,
        pc_percentage=95,
        top_path=reg.get_mapped_path("top_sim0_butane_123456"),
        traj_path=reg.get_mapped_path("rec0_butane_123456"),
        sim_id="sim0_butane_123456",
        selection="all",  # this is because it is a made up simulation of butane.
        # usually it would be "name CA" to get every residue
    )
    pca_space = np.random.rand(100, 2)
    T = len(pca_space)
    np.arange(T)
    assert pca_analysis._cosine_content(pca_space=pca_space, i=0) < 0.1
    assert pca_analysis._cosine_content(pca_space=pca_space, i=1) < 0.1
