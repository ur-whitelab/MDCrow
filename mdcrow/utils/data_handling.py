import os
import warnings

import matplotlib.pyplot as plt
import mdtraj as md
import numpy as np

from .path_registry import FileType, PathRegistry


def load_single_traj(
    path_registry,
    top_fileid=None,
    traj_fileid=None,
    traj_required=False,
    ignore_warnings=False,
):
    """
    Load a single trajectory file using mdtraj. Check for file IDs in the path registry.

    Parameters:
    path_registry (PathRegistry): mapping file IDs to file paths.
    top_fileid (str, optional): File ID for the topology file.
    traj_fileid (str, optional): File ID for the trajectory file.
    traj_required (bool, optional): Whether the traj file is required. Default is False.

    Returns:
    mdtraj.Trajectory: Trajectory object.
    """
    if not isinstance(path_registry, PathRegistry):
        raise ValueError("path_registry must be an instance of PathRegistry.")
    all_fileids = path_registry.list_path_names()

    if top_fileid is None:
        # check for traj_fileid
        if traj_fileid is None:
            raise ValueError("Trajectory File ID is required, and it's not provided.")
    else:
        if top_fileid not in all_fileids:
            raise ValueError(
                f"Topology File ID '{top_fileid}' not found in PathRegistry"
            )
        else:
            top_path = path_registry.get_mapped_path(top_fileid)

    if traj_fileid is None:
        if not traj_required:
            if not ignore_warnings:
                warnings.warn(
                    (
                        "Trajectory File ID is not provided but is not required; "
                        f"loading MDTrajectory from topology {top_fileid} only."
                    ),
                    UserWarning,
                )

            return md.load(top_path)
        else:
            raise ValueError("Trajectory File ID is required, and it's not provided.")
    else:
        if traj_fileid not in all_fileids:
            raise ValueError(
                f"Trajectory File ID '{traj_fileid}' not found in PathRegistry."
            )
        else:
            traj_path = path_registry.get_mapped_path(traj_fileid)
    if top_fileid is None:
        return md.load(traj_path)
    elif traj_fileid is None:
        return md.load(top_path)
    return md.load(traj_path, top=top_path)


def load_traj_with_ref(
    path_registry,
    top_id,
    traj_id=None,
    ref_top_id=None,
    ref_traj_id=None,
    traj_required=False,
    ignore_warnings=False,
):
    traj = load_single_traj(
        path_registry, top_id, traj_id, traj_required, ignore_warnings
    )
    if ref_top_id is None:
        ref_traj = traj
    else:
        ref_traj = load_single_traj(
            path_registry, ref_top_id, ref_traj_id, traj_required, ignore_warnings
        )

    return traj, ref_traj


def save_to_csv(
    path_registry, data_to_save, analysis_name, description=None, header=""
):
    """
    Saves data to a csv file and maps the file ID to the file path in the path registry.

    Parameters:
    path_registry (PathRegistry): mapping file IDs to file paths.
    data_to_save (np.ndarray): Data to save to a csv file.
    analysis_name (str): Name of the analysis or data. This will be used as the file ID.
    description (str, optional): Description of the data.
    header (str, optional): Header for the csv file.

    Returns:
    str: File ID for the saved data.
    """
    if not isinstance(path_registry, PathRegistry):
        raise ValueError("path_registry must be an instance of PathRegistry.")
    if not isinstance(data_to_save, np.ndarray):
        raise TypeError("data_to_save must be an instance of np.ndarray.")

    base_path = f"{path_registry.ckpt_records}/{analysis_name}"
    file_path = f"{base_path}.csv"
    i = 0
    while os.path.exists(file_path):
        i += 1
        file_path = f"{base_path}_{i}.csv"
    file_id = analysis_name if i == 0 else f"{analysis_name}_{i}"
    np.savetxt(file_path, data_to_save, delimiter=",", header=header)
    path_registry.map_path(file_id, file_path, description=description)
    print(f"Data saved to {file_path}")
    return file_id


def save_plot(path_registry, fig_analysis, description=None):
    if not isinstance(path_registry, PathRegistry):
        raise ValueError("path_registry must be an instance of PathRegistry.")
    if plt.gcf().get_axes() == []:  # if there's no plot
        raise ValueError("No plot detected. Failed to save.")

    fig_name = path_registry.write_file_name(
        type=FileType.FIGURE,
        fig_analysis=fig_analysis,
        file_format="png",
    )
    fig_id = path_registry.get_fileid(file_name=fig_name, type=FileType.FIGURE)
    fig_path = f"{path_registry.ckpt_figures}/{fig_name}"
    plt.savefig(fig_path)
    path_registry.map_path(fig_id, fig_path, description=description)
    print(f"Plot saved to {fig_path}")
    return fig_id
