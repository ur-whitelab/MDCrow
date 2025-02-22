from unittest.mock import patch

from mdcrow.tools.base_tools.analysis_tools.bond_angles_dihedrals_tool import (
    ComputeAngles,
)


def test_compute_angles_tool_bad_inputs(get_registry):
    reg = get_registry("raw", True, map_path=True, include_peptide_trajectory=True)
    angles_tool = ComputeAngles(path_registry=reg)
    bad_input_files = {
        "trajectory_fileid": "pep_traj_987654_3",
        "topology_fileid": "pep_traj_987654_3",
        "analysis": "both",
    }

    error_catching = angles_tool._run(bad_input_files)
    assert "Trajectory File ID not in path registry" in error_catching
    assert "Topology File ID not in path registry" in error_catching


def test_compute_angles_ram_values(get_registry):
    reg = get_registry("raw", True, dynamic=True, include_hydrogens=True)
    angles_tool = ComputeAngles(path_registry=reg)
    phi_psi_input_files = {
        "trajectory_fileid": "pep_traj_987654",
        "topology_fileid": "pep_traj_987654",
        "analysis": "phi-psi",
    }
    chi_innput_files = {
        "trajectory_fileid": "pep_traj_987654",
        "topology_fileid": "pep_traj_987654",
        "analysis": "chi1-chi2",
    }

    with patch(
        "mdcrow.tools.base_tools.analysis_tools.ComputeAngles.compute_and_plot_phi_psi"
    ) as mock_compute_and_plot_phi_psi:
        with patch(
            "mdcrow.tools.base_tools.analysis_tools.ComputeAngles.compute_plot_all_chi_angles"
        ) as compute_plot_all_chi_angles:
            mock_compute_and_plot_phi_psi.return_value = ("mockid", "mockresult")
            # instance.return_value = ("mockid", "mockresult")
            angles_tool._run(phi_psi_input_files)
            # print(result)
            assert mock_compute_and_plot_phi_psi.called
            # assert compute_and_plot_chi1_chi2 is not called
            assert not compute_plot_all_chi_angles.called

            # =========================================================================#
            compute_plot_all_chi_angles.return_value = ("mockid", "mockresult")
            angles_tool._run(chi_innput_files)
            assert compute_plot_all_chi_angles.called
            # assert compute_and_plot_phi_psi is not called
            assert (
                mock_compute_and_plot_phi_psi.assert_called_once
            )  # already called once
