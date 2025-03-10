from typing import Optional

import mdtraj as md
import numpy as np
from state import MDCrowState
from utils import FileType, PathRegistry, load_single_traj


def _summarize_protein_structure(
    traj, requested_analyses: list | None = None
) -> dict[str, int]:
    """
    Summarizes the structure of a protein trajectory.

    Args:
        traj: The trajectory to summarize the
            structure of.
        requested_analyses: A list of the analyses
            to include in the summary.

    Returns:
        A dictionary containing the requested analyses.
    """
    if not traj.topology:
        raise ValueError("Topolgy is required for this analysis to be meaningful.")
    if not requested_analyses:
        requested_analyses = ["atoms", "residues", "chains", "frames", "bonds"]
    result = {}
    if "atoms" in requested_analyses:
        result["n_atoms"] = traj.n_atoms
    if "residues" in requested_analyses:
        result["n_residues"] = traj.n_residues
    if "chains" in requested_analyses:
        result["n_chains"] = traj.n_chains
    if "frames" in requested_analyses:
        result["n_frames"] = traj.n_frames
    if "bonds" in requested_analyses:
        result["n_bonds"] = len([bond for bond in traj.topology.bonds])
    return result


async def summarize_protein_structure(
    state: MDCrowState,
    traj_file: str,
    top_file: Optional[str] = None,
    requested_analyses: Optional[list[str]] = None,
):
    """
        Get the number of atoms, residues, chains, frames, and bonds in a \
        protein trajectory. Input is a trajectory file ID and an optional topology \
        file ID. The output is a dictionary containing the analyses.
        Args:
            traj_fileid (str): The trajectory to summarize the
                structure.
            top_fileid (str): The topology file for the trajectory.
            requested_analyses Optional[list[str]]: A list of the analyses
                to include in the summary.

        Returns:
            A dictionary containing the requested analyses.
    """
    try:
        traj = load_single_traj(
            path_registry=state.path_registry,
            traj_fileid=traj_file,
            top_fileid=top_file,
        )
        if not traj:
            raise Exception("Trajectory could not be loaded.")
    except Exception as e:
        return str(e), 0, False
    try:
        result = _summarize_protein_structure(
            traj, requested_analyses=requested_analyses
        )
    except Exception as e:
        return str(e), 0, False
    return str(result), 0, False


def _dssp_codes(simplified) -> list[str]:
    """
    Returns the DSSP codes used by MDTraj. If simplified is True, only
    the codes H, E, and C are used. Otherwise, the full set of codes is
    used."""
    if simplified:
        return ["H", "E", "C", "NA"]
    return ["H", "B", "E", "G", "I", "T", "S", " ", "NA"]


def _dssp_natural_language(simplified: bool = False) -> dict[str, str]:
    """
    Returns a dictionary mapping DSSP codes to their natural language
    descriptions. If simplified is True, only the codes H, E, and C are
    used. Otherwise, the full set of codes is used."""
    if simplified:
        return {
            "H": "residues in helix",
            "E": "residues in strand",
            "C": "residues in coil",
            "NA": "residues not assigned, not a protein residue",
        }
    return {
        "H": "residues in alpha helix",
        "B": "residues in beta bridge",
        "E": "residues in extended strand",
        "G": "residues in three helix",
        "I": "residues in five helix",
        "T": "residues in hydrogen bonded turn",
        "S": "residues in bend",
        " ": "residues in loop or irregular",
        "NA": "residues not assigned, not a protein residue",
    }


def _convert_dssp_counts(dssp_counts: dict, simplified: bool = False) -> dict:
    """
    Converts a dictionary of DSSP codes to their counts into a dictionary
    of natural language descriptions to their counts.
    is used.

    Args:
        dssp_counts: A dictionary mapping DSSP codes to their counts.

    Returns:
        A dictionary mapping natural language descriptions to their counts.
    """
    code_to_description = _dssp_natural_language(simplified=simplified)

    descriptive_counts = {
        code_to_description[code]: count for code, count in dssp_counts.items()
    }
    return descriptive_counts


def _summarize_dssp(dssp_array: np.ndarray, simplified: bool = False) -> dict[str, int]:
    """
    Summarizes the DSSP assignments for a trajectory. Returns a dictionary
    mapping DSSP codes to their counts.

    Args:
        dssp_array: An array of DSSP codes for each residue at each time point.

    Returns:
        A dictionary mapping natural language descriptions to their counts.
    """
    dssp_codes = _dssp_codes(simplified=simplified)
    dssp_dict = {code: 0 for code in dssp_codes}
    for frame in dssp_array:
        for code in frame:
            if code in dssp_dict.keys():
                dssp_dict[code] += 1
            else:
                dssp_dict[code] = 1
    return _convert_dssp_counts(dssp_dict, simplified=simplified)


def _compute_dssp(traj: md.Trajectory, simplified: bool = False) -> np.ndarray:
    """
    Computes the DSSP assignments for a trajectory.

    Args:
        traj: The trajectory to compute DSSP assignments for.

    Returns:
        An array of DSSP codes for each residue at each time point.
    """
    return md.compute_dssp(traj, simplified=simplified)


def _get_frame(traj, target_frames):
    """
    Retrieves the target frame(s) of the trajectory for DSSP.

    Args:
        traj: the trajectory
        target_frames: the target frames to select. can be first, last, or all

    Returns:
        the trajectory with only target frames"""

    if target_frames.lower().strip() == "all":
        return traj
    if target_frames.lower().strip() == "first":
        return traj[0]
    if target_frames.lower().strip() == "last":
        return traj[-1]
    else:
        raise ValueError("Target Frames must be 'all', 'first', or 'last'.")


def write_raw_x(
    x: str, values: np.ndarray, traj_id: str, path_registry: PathRegistry
) -> str:
    """
    Writes raw x values to a file and saves the file to the path registry.

    Args:
        x: The name of the analysis tool that produced the values (e.g., "dssp")
        values: The x values to save.
        traj_id: The id of the trajectory the values are associated with.
        path_registry: The path registry to save the file to.

        Returns:
            The file id of the saved file.
    """
    file_name = path_registry.write_file_name(
        FileType.RECORD, record_type=x, file_format="npy"
    )
    file_id = path_registry.get_fileid(file_name, FileType.RECORD)

    file_path = f"{path_registry.ckpt_records}/{file_name}"
    np.save(file_path, values)

    path_registry.map_path(
        file_id,
        file_path,
        description=f"{x} values for trajectory with id: {traj_id}",
    )
    return file_id


async def computeDSSP(
    state: MDCrowState,
    traj_file: str,
    top_file: Optional[str] = None,
    target_frames: str = "last",
    simplified: bool = True,
):
    """Compute the DSSP (secondary structure) assignment
    for a protein trajectory. Input is a trajectory file ID and
    a target_frames, which can be "first", "last", or "all",
    and an optional topology file ID.
    Input "first" to get DSSP of only the first frame.
    Input "last" to get DSSP of only the last frame.
    Input "all" to get DSSP of all frames in trajectory, combined.
    The output is an array with the DSSP code for each
    residue at each time point.

    Args:
        traj_file (str): The trajectory file ID.
        top_file (str): The topology file ID.
        target_frames (str): The target frames to select.
        simplified (bool): If True, only the codes H, E, and C are used.
            Otherwise, the full set of codes is used.
    """

    try:
        traj = load_single_traj(
            path_registry=state.path_registry,
            traj_fileid=traj_file,
            top_fileid=top_file,
        )
        if not traj:
            raise Exception("Trajectory could not be loaded.")
        traj = _get_frame(traj, target_frames)
    except Exception as e:
        print("Error loading trajectory: ", e)
        return str(e), 0, False

    dssp_array = _compute_dssp(traj, simplified=simplified)
    write_raw_x("dssp", dssp_array, traj_file, state.path_registry)
    summary = _summarize_dssp(dssp_array, simplified=simplified)
    return str(summary), 0, False
