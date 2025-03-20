import asyncio
from unittest.mock import patch

from mdcrow.ldp_env.analysis_tools.bond_angles_and_dihedrals import (
    compute_bond_angles,
    validate_input,
)
from mdcrow.ldp_env.state import MDCrowState


def test_compute_angles_tool_bad_inputs(get_registry):
    reg = get_registry("raw", True, map_path=True, include_peptide_trajectory=True)
    state = MDCrowState(path_registry=reg, tools=[])
    bad_input_files = {
        "trajectory_fileid": "pep_traj_987654_3",
        "topology_fileid": "pep_traj_987654_3",
        "analysis": "both",
    }

    error_catching = validate_input(state.path_registry, **bad_input_files)
    # get error messages
    error_message = error_catching.get("error")
    assert "Trajectory File ID not in path registry" in error_message
    assert "Topology File ID not in path registry" in error_message


def test_compute_angles_ram_values(get_registry):
    reg = get_registry("raw", True, dynamic=True, include_hydrogens=True)
    state = MDCrowState(path_registry=reg, tools=[])

    phi_psi_input_files = {
        "trajectory_fileid": "pep_traj_987654",
        "topology_fileid": "pep_traj_987654",
        "analysis": "phi-psi",
    }
    chi_input_files = {
        "trajectory_fileid": "pep_traj_987654",
        "topology_fileid": "pep_traj_987654",
        "analysis": "chis",  # Updated to match valid options in new implementation
    }

    with patch(
        "mdcrow.ldp_env.analysis_tools.bond_angles_and_dihedrals.compute_and_plot_phi_psi"
    ) as mock_compute_and_plot_phi_psi:
        with patch(
            "mdcrow.ldp_env.analysis_tools.bond_angles_and_dihedrals.compute_plot_all_chi_angles"
        ) as mock_compute_plot_all_chi_angles:

            mock_compute_and_plot_phi_psi.return_value = ("mockid", "mockresult")

            # Call compute_bond_angles with phi-psi input
            asyncio.run(compute_bond_angles(state, **phi_psi_input_files))

            # Verify that compute_and_plot_phi_psi was called
            assert mock_compute_and_plot_phi_psi.called
            assert not mock_compute_plot_all_chi_angles.called  # Ensure chi analysis
            # wasn't called
            # =========================================================================#
            mock_compute_plot_all_chi_angles.return_value = ("mockid", "mockresult")

            # Call compute_bond_angles with chi input
            asyncio.run(compute_bond_angles(state, **chi_input_files))

            assert mock_compute_plot_all_chi_angles.called
            assert mock_compute_and_plot_phi_psi.call_count == 1  # Should still
            # be called only once
