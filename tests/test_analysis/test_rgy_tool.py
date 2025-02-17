import pytest

from mdcrow.tools.base_tools.analysis_tools.rgy import (
    RadiusofGyration,
    RadiusofGyrationTool,
)


@pytest.fixture
def rgy(get_registry, loaded_cif_traj):
    registry = get_registry("raw", False)
    rgy = RadiusofGyration(path_registry=registry)
    rgy.traj = loaded_cif_traj
    rgy.top_file = "test_top_dummy"
    rgy.traj_file = "test_traj_dummy"
    return rgy


@pytest.fixture
def rgy_tool(get_registry, loaded_cif_traj):
    registry = get_registry("raw", False)
    rgy_tool = RadiusofGyrationTool(path_registry=registry, load_traj=False)
    rgy_tool.rgy.traj = loaded_cif_traj
    rgy_tool.rgy.top_file = "test_top_dummy"
    rgy_tool.rgy.traj_file = "test_traj_dummy"
    return rgy_tool


def test_rgy_tool(rgy_tool):
    output = rgy_tool._run(traj_file="test_top_dummy", top_file="test_traj_dummy")
    assert "Radii of gyration saved to " in output
    assert "Average radius of gyration: " in output
    assert "Plot saved as: " in output
    assert ".png" in output


def test_rgy_per_frame(rgy):
    output = rgy.rgy_per_frame()
    assert "Radii of gyration saved to " in output


def test_rgy_average(rgy):
    output = rgy.rgy_average()
    assert "Average radius of gyration: " in output


def test_plot_rgy(rgy):
    output = rgy.plot_rgy()
    assert "Plot saved as: " in output
    assert ".png" in output


def test_compute_plot_return_avg_rgy(rgy):
    output = rgy.compute_plot_return_avg()
    assert "Radii of gyration saved to " in output
    assert "Average radius of gyration: " in output
    assert "Plot saved as: " in output
    assert ".png" in output
