from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pytest

from mdcrow.utils import load_single_traj, load_traj_with_ref, save_plot, save_to_csv


@pytest.fixture
def load_single_traj_mock():
    with patch("mdcrow.utils.data_handling.load_single_traj", return_value="MockTraj"):
        yield


# simple helper functions for tools
def test_load_traj_only_topology(get_registry):
    registry = get_registry("raw", True)
    with patch.object(
        registry, "get_mapped_path", wraps=registry.get_mapped_path
    ) as mocked_get_mapped_path:
        traj = load_single_traj(
            registry, "top_sim0_butane_123456", ignore_warnings=True
        )
        mocked_get_mapped_path.assert_called_once()
        assert traj is not None


def test_load_traj_topology_and_traj(get_registry):
    registry = get_registry("raw", True)
    with patch.object(
        registry, "get_mapped_path", wraps=registry.get_mapped_path
    ) as mocked_get_mapped_path:
        traj = load_single_traj(
            registry, "top_sim0_butane_123456", "rec0_butane_123456"
        )
        assert mocked_get_mapped_path.call_count == 2
        assert traj is not None


def test_load_traj_fail_top_fileid(get_registry):
    registry = get_registry("raw", False)
    with pytest.raises(ValueError) as exc:
        load_single_traj(registry, "top_invalid", ignore_warnings=True)
    assert "Topology File ID 'top_invalid' not found" in str(exc.value)


def test_load_traj_fail_traj_fileid(get_registry):
    registry = get_registry("raw", True)
    with pytest.raises(ValueError) as exc:
        load_single_traj(registry, "top_sim0_butane_123456", "traj_invalid")
    assert "Trajectory File ID 'traj_invalid' not found" in str(exc.value)


def test_load_traj_with_ref_both(get_registry, load_single_traj_mock):
    path_registry = get_registry("raw", False)
    traj, ref_traj = load_traj_with_ref(
        path_registry, "top_id", "traj_id", "ref_top_id", "ref_traj_id"
    )
    assert traj == "MockTraj"
    assert ref_traj == "MockTraj"


def test_load_traj_with_ref_only_toplogy(get_registry, load_single_traj_mock):
    path_registry = get_registry("raw", False)
    traj, ref_traj = load_traj_with_ref(path_registry, "top_id", ignore_warnings=True)
    assert traj == "MockTraj"
    assert ref_traj == "MockTraj"


def test_save_plot_success(get_registry):
    path_registry = get_registry("raw", False)
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 2, 3])  # Create a plot
    with patch("mdcrow.utils.data_handling.plt.savefig"):
        fig_id = save_plot(path_registry, "test_data", "Test plot")
        assert "fig0_" in fig_id


def test_save_plot_no_plot_detected(get_registry):
    plt.close("all")  # clear any past plots
    path_registry = get_registry("raw", False)
    with pytest.raises(ValueError) as excinfo:
        save_plot(path_registry, "test_analysis")
    assert "No plot detected" in str(excinfo.value)


def test_save_plot_wrong_registry_type():
    with pytest.raises(ValueError) as excinfo:
        save_plot(None, "test_analysis")
    assert "path_registry must be an instance" in str(excinfo.value)


def test_save_to_csv(get_registry):
    registry = get_registry("raw", False)
    data = np.array([[1, 2], [3, 4]])
    with patch("os.path.exists", return_value=False):
        with patch("numpy.savetxt", return_value=None):
            csv_path = save_to_csv(registry, data, "test_id", "Description of data")
            assert "test_id" in csv_path
