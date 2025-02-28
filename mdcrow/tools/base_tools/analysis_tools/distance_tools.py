import itertools
import os
from typing import Literal, Optional

import matplotlib.pyplot as plt
import mdtraj as md
import numpy as np
import pandas as pd
from langchain.tools import BaseTool
from matplotlib.animation import FuncAnimation
from pydantic import BaseModel, Field

from mdcrow.utils import FileType, PathRegistry, load_single_traj


class DistanceToolsUtils:
    def __init__(self, path_registry: Optional[PathRegistry] = None):
        self.path_registry = path_registry

    def all_possible_pairs(self, array1, array2):
        return list(itertools.product(array1, array2))

    def calc_side_center_mass(self, traj):
        """
        Compute approximate center of mass of each side chain per frame.
        COM od Glycine is approximated by the coordinate of its CA atom.

        Note: Because compute_center_of_mass gets the center of mass of only one
        selection at a time, we need to loop over all residues to get the each COM.

        returns: com_matrix: np.array, shape=(n_frames, n_residues, 3)
        """
        traj = traj.atom_slice(traj.top.select("protein"), inplace=False)
        com_matrix = np.zeros((traj.n_frames, traj.n_residues, 3))
        for i in range(0, traj.topology.n_residues):
            selection = f"(resid {i}) and sidechain"
            if len(selection) < 1:
                selection = f"(resid {i}) and (name CA)"
            try:
                com_matrix[:, i, :] = md.compute_center_of_mass(traj, select=selection)
            except Exception as e:
                raise (f"Error calculating center of mass for residue, {i}: {str(e)}")
        return com_matrix

    def calc_dis_matrix_from_com_all_resids(self, traj, com_matrix):
        """
        Compute the distance matrix between the center of mass of residues

        returns: dis_matrix: np.array, shape=(n_frames, n_residues, n_residues)
        """
        new_xyz = com_matrix
        new_topology = pd.DataFrame(
            {
                "serial": range(traj.n_residues),
                "name": ["COM" for _ in traj.topology.residues],
                "resSeq": range(traj.n_residues),
                "resName": [i.name for i in traj.topology.residues],
                "element": ["VS" for _ in traj.topology.residues],
                "chainID": [
                    i.chain.chain_id if i.chain.chain_id else 0
                    for i in traj.topology.residues
                ],
                "segmentID": [i.segment_id for i in traj.topology.residues],
            }
        )
        top = md.Topology.from_dataframe(new_topology)
        new_traj = md.Trajectory(new_xyz, top)
        pairs = self.all_possible_pairs(range(traj.n_residues), range(traj.n_residues))
        _dis_matrix = md.compute_distances(new_traj, pairs)
        dis_matrix = md.geometry.squareform(_dis_matrix, residue_pairs=pairs)
        return dis_matrix

    def calc_matrix_cm_all_resids(self, traj, threshold=0.8, distance=1.2):
        """
        Used internally to compute the matrix of contacts between residues

        returns: matrix: np.array, shape=(n_frames, n_residues, n_residues)
        """

        # By default, this ↓ ignores any non-protein atoms
        distances, residue_pairs = md.compute_contacts(traj, scheme="closest-heavy")
        matrix = md.geometry.squareform(distances, residue_pairs=residue_pairs)
        for frame in range(matrix.shape[0]):
            for i in range(matrix.shape[1]):
                for j in range(matrix.shape[2]):
                    if matrix.shape[0] == matrix.shape[1]:
                        if i > j:
                            if matrix[frame, i, j] < threshold:
                                matrix[frame, i, j] = 1
                            else:
                                matrix[frame, i, j] = 0
                        else:
                            if matrix[frame, i, j] < threshold:
                                matrix[frame, i, j] = 1
                            elif matrix[frame, i, j] > distance:
                                matrix[frame, i, j] = 0
                            else:
                                matrix[frame, i, j] = 1 - (
                                    matrix[frame, i, j] - threshold
                                ) / (distance - threshold)
                    else:
                        return matrix < threshold
        return matrix

    def calc_matrix_dis_ca_all_resids(self, traj):
        """
        Gets the Matrix distance between all alpha carbons of residues
        returns: dis_matrix: np.array, shape=(n_frames, n_residues, n_residues)
        """
        self.calc_side_center_mass(traj)
        traj = traj.atom_slice(traj.top.select("protein"))
        residues = traj.topology.select("name CA")
        residues_ids = np.arange(0, traj.topology.n_residues)
        residue_pairs = self.all_possible_pairs(residues, residues)
        # needed to use squareform!
        residue_id_pairs = self.all_possible_pairs(residues_ids, residues_ids)
        distances = md.compute_distances(traj, residue_pairs)
        dis_matrix = md.geometry.squareform(distances, residue_pairs=residue_id_pairs)

        return dis_matrix

    def save_matrix_frame(self, matrix, path):
        """
        Saves the distance matrix as images in the path specified. It samples 10 frames
        from the matrix and saves them as images.
        """
        if not os.path.exists(path):
            os.makedirs(path)

        for i in range(0, matrix.shape[0], matrix.shape[0] // 10):
            fig, ax = plt.subplots()
            ax.imshow(matrix[i], origin="lower")
            ax.set_title(f"Distance Matrix frame {i}")

            plt.savefig(f"{path}/dist_matrix_frame_{i}.png")

    def make_movie(
        self, matrix, option="distance", source_id="unkown", path="distance.gif"
    ):
        """
        Create an animation of the distance matrix over time
        saves: a gif of the animation
        returns: Description of the animation
        """
        Title = option.capitalize() + "Matrix"
        fig, ax = plt.subplots()
        im = ax.imshow(matrix[0], origin="lower")

        # Create a title text object
        title_text = ax.set_title(f"{Title} frame {0}")

        def update(frame):
            im.set_array(matrix[frame])
            # Update the title
            title_text.set_text(f"{Title} frame {frame}")
            return im, title_text  # Return both the image and the title text object

        ani = FuncAnimation(fig, update, frames=matrix.shape[0], interval=200)
        ani.save(path)
        description = f"{option.capitalize()} matrix over time with {matrix.shape[0]} \
                frames, and {matrix.shape[1]} residues. \
                trajectory file: {source_id}"
        return description


class DistanceSchema(BaseModel):
    trajectory_fileid: str = Field(
        description="Trajectory File ID of the simulation to be analyzed"
    )
    topology_fileid: str = Field(
        description="Topology File ID of the simulation to be analyzed"
    )
    analysis: Literal["all", "not all"] = Field(
        "all",
        description=(
            "Which residues ids to calculate distance from, if all, "
            "all residues will be used if not all, only the selected"
            " residues (selection1 and selection2) will be used"
        ),
    )
    mode: Literal["CA", "COM"] = Field(
        "CA",
        description=(
            "What to use for distance calculation, either "
            "alpha carbons (CA) or center of mass (COM)"
        ),
    )
    selection1: Optional[str] = Field(
        description="First"
        + (
            "Selection of residues ids from the simulation to use for the analysis."
            "Example selection: 'resid 0 to 10' or 'resid 0 1 2 3 4 5 6 7 8 9 10'"
        )
    )
    selection2: Optional[str] = Field(
        description="Second"
        + (
            "Selection of residues ids from the simulation to use for the analysis."
            "Example selection: 'resid 0 to 10' or 'resid 0 1 2 3 4 5 6 7 8 9 10'"
        )
    )


class ContactSchema(BaseModel):
    trajectory_fileid: str = Field(
        description="Trajectory File ID of the simulation to be analyzed"
    )
    topology_fileid: str = Field(
        description="Topology File ID of the simulation to be analyzed"
    )
    selection: str = Field(
        description="Selection of residues from the \
                        simulation to use for the contact analysis. Default is 'all'\
                        which will calculate the distance between all residue pairs.\
                        \nExample selection: 'resid 0 to 10' or \
                        'resid 0 1 2 3 4 5 6 7 8 9 10' or 'all'"
    )
    cutoff: float = Field(
        0.8,
        description="Hard cutoff distance for the contact  \
                          analysis in nanometers. Defaults to 0.8",
    )


class DistanceMatrixTool(BaseTool):
    name = "DistanceMatrixTool"
    description = (
        "Tool for calculating distances between residue pairs in each frame of a "
        "trajectory. If only one pair is provided, the tool will calculate the distance"
        " between said pair in each frame and output a distance vs time plot and a "
        "histogram. If multiple pairs are provided, the tool will calculate the "
        "distance between each pair in each frame and output a distance matrix plot "
        "for the selected pairs.\n You can use 'analysis' = 'all' to calculate the "
        "distance between all residue pairs in each frame. Or if interested in a "
        "specific pair, you can provide two selections of residues/atoms to calculate "
        "the distance between them."
    )
    input_schema = DistanceSchema
    path_registry: Optional[PathRegistry]

    def __init__(self, path_registry: Optional[PathRegistry] = None):
        super().__init__()
        self.path_registry = path_registry

    def _run(self, **input):
        try:
            input = self.validate_inputs(**input)
        except ValueError as e:
            return f"Error using the PCA Tool: {str(e)}"

        (
            trajectory_id,
            topology_id,
            selection1,
            selection2,
            analysis,
            mode,
            error,
            system_message,
        ) = self.get_values(input)

        if error:
            return f"Failed. Error with the tool inputs: {error} "

        try:
            traj = load_single_traj(
                self.path_registry,
                topology_id,
                traj_fileid=trajectory_id,
                traj_required=True,
            )
        except ValueError as e:
            if (
                "The topology and the trajectory files might not\
                  contain the same atoms"
                in str(e)
            ):
                return (
                    "Failed. Error loading trajectory. Make sure the topology file"
                    " is from the initial positions of the trajectory. Error: {str(e)}"
                )
            return f"Failed. Error loading trajectory: {str(e)}"
        except OSError as e:
            if (
                "The topology is loaded by filename extension, \
                and the detected"
                in str(e)
            ):
                return (
                    "Failed. Error loading trajectory. Make sure you include the"
                    "correct file for the topology. Supported extensions are:"
                    "'.pdb', '.pdb.gz', '.h5', '.lh5', '.prmtop', '.parm7', '.prm7',"
                    "  '.psf', '.mol2', '.hoomdxml', '.gro', '.arc', '.hdf5' and '.gsd'"
                )
            return f"Failed. Error loading trajectory: {str(e)}"
        except Exception as e:
            return f"Failed. Error loading trajectory: {str(e)}"

        if analysis != "all":
            # slice the trajectory to include only the selections
            atom_indices1 = traj.top.select(selection1)
            atom_indices2 = traj.top.select(selection2)
            traj = traj.atom_slice(atom_indices1 + atom_indices2, inplace=False)
        utils = DistanceToolsUtils(path_registry=self.path_registry)
        if mode == "CA":
            dist_matrix = utils.calc_matrix_dis_ca_all_resids(traj)
        elif mode == "COM":
            # calculates distances matrix using center of mass of side chains
            com_matrix = utils.calc_side_center_mass(traj)
            dist_matrix = utils.calc_dis_matrix_from_com_all_resids(traj, com_matrix)

        # plotting the distance matrix
        path = f"{self.path_registry.ckpt_figures}/dist_{trajectory_id}/\
            dist_matrix_{trajectory_id}.gif"
        if not os.path.exists(
            f"{self.path_registry.ckpt_figures}/dist_{trajectory_id}"
        ):
            os.makedirs(f"{self.path_registry.ckpt_figures}/dist_{trajectory_id}")
        fig_id = self.path_registry.get_fileid(file_name=path, type=FileType.FIGURE)
        movie_desc = utils.make_movie(dist_matrix, path=path, source_id=trajectory_id)
        self.path_registry.map_path(fig_id, path, movie_desc)
        # save some of the frames as images
        utils.save_matrix_frame(
            dist_matrix, f"{self.path_registry.ckpt_figures}/dist_{trajectory_id}"
        )

        return "Succeeded. Distance Matrix created with ID: " + fig_id + system_message

    def _arun(self):
        pass

    def validate_inputs(self, input):
        input = input.get("action_input", input)
        input = input.get("input", input)
        trajectory_id = input.get("trajectory_fileid", None)
        topology_id = input.get("topology_fileid", None)
        selection1 = input.get("selection1", "name CA")
        selection2 = input.get("selection2", "name CA")
        mode = input.get("mode", "CA")
        analysis = input.get("analysis", "all")
        if not trajectory_id:
            raise ValueError("Incorrect Inputs: trajectory_fileid is required")
        if not topology_id:
            raise ValueError("Incorrect Inputs: topology_fileid is required")
        fileids = self.path_registry.list_path_names()
        error = ""
        system_message = ""
        if trajectory_id not in fileids:
            error += " Trajectory File ID not in path registry"
        if topology_id not in fileids:
            error += " Topology File ID not in path registry"
        if mode not in ["CA", "COM"]:
            print("Incorrect mode, must be either CA or COM. Defaulting to CA")
            system_message += " 'mode' must be either CA or COM. \
                Tool defaulted to measure distances w.r.t. alpha carbons (CA) \n"
            mode = "CA"
        if analysis not in ["all", "not all"]:
            print(
                "Incorrect analysis, must be either 'all' or 'not all'."
                " Defaulting to all"
            )
            system_message += "'analysis', must be either 'all' or 'not all'.\
                  Tool defaulted to 'all' \n"
            analysis = "all"
        keys = input.keys()
        for key in keys:
            if key not in [
                "trajectory_fileid",
                "topology_fileid",
                "analysis",
                "mode",
                "selection1",
                "selection2",
            ]:
                system_message += (
                    f"{key} is not part of admitted tool inputs\n."
                    "Ignoring it during the analysis"
                )

        if error == "":
            error = None
        return {
            "trajectory_fileid": trajectory_id,
            "topology_fileid": topology_id,
            "selection1": selection1,
            "selection2": selection2,
            "analysis": analysis,
            "mode": mode,
            "error": error,
            "system_message": system_message,
        }

    def get_values(self, input):
        traj_id = input.get("trajectory_fileid")
        top_id = input.get("topology_fileid")
        sel1 = input.get("selection1")
        sel2 = input.get("selection2")
        analysis = input.get("analysis")
        mode = input.get("mode")
        error = input.get("error")
        syst_mes = input.get("system_message")

        return traj_id, top_id, sel1, sel2, analysis, mode, error, syst_mes


class ContactsTool(BaseTool):
    name = "ContactsTool"
    description = (
        "Tool for computing the distance between pairs of residues in a trajectory. "
        "If distance is under the cutoff is considered a contact. The output is a "
        "matrix plot where each contact between residues is represented by a dot."
    )
    input_schema = ContactSchema
    path_registry: Optional[PathRegistry]

    def __init__(self, path_registry: Optional[PathRegistry] = None):
        super().__init__()
        self.path_registry = path_registry

    def _run(self, **input):
        try:
            input = self.validate_input(**input)
        except ValueError as e:
            return f"Failed. Error using the Contacts Tool: {str(e)}"
        (
            trajectory_id,
            topology_id,
            selection,
            cutoff,
            error,
            system_message,
        ) = self.get_values(input)

        if error:
            return f"Failed. Error with the tool inputs: {error} "

        try:
            traj = load_single_traj(
                self.path_registry,
                topology_id,
                traj_fileid=trajectory_id,
                traj_required=True,
            )
        except ValueError as e:
            if (
                "The topology and the trajectory files might not\
                  contain the same atoms"
                in str(e)
            ):
                return (
                    "Failed. Error loading trajectory. Make sure the topology file"
                    " is from the initial positions of the trajectory. Error: {str(e)}"
                )
            return f"Failed. Error loading trajectory: {str(e)}"
        except OSError as e:
            if (
                "The topology is loaded by filename extension, \
                and the detected"
                in str(e)
            ):
                return (
                    "Failed. Error loading trajectory. Make sure you include the"
                    "correct file for the topology. Supported extensions are:"
                    "'.pdb', '.pdb.gz', '.h5', '.lh5', '.prmtop', '.parm7', '.prm7',"
                    "  '.psf', '.mol2', '.hoomdxml', '.gro', '.arc', '.hdf5' and '.gsd'"
                )
            return f"Failed. Error loading trajectory: {str(e)}"
        except Exception as e:
            return f"Failed. Error loading trajectory: {str(e)}"

        if selection != "all":
            try:
                atom_indices = traj.top.select(selection)
                traj = traj.atom_slice(atom_indices, inplace=False)
            except Exception as e:
                system_message += f"Error with the selection: {str(e)}.\
                      Defaulting to 'all'"

        utils = DistanceToolsUtils(path_registry=self.path_registry)
        matrix = utils.calc_matrix_cm_all_resids(traj, threshold=cutoff)

        # save the matrix as a gif and some of the frames as images
        path = f"{self.path_registry.ckpt_figures}/contact_{trajectory_id}/\
            contact_matrix_{trajectory_id}.gif"
        if not os.path.exists(
            f"{self.path_registry.ckpt_figures}/contact_{trajectory_id}"
        ):
            os.makedirs(f"{self.path_registry.ckpt_figures}/contact_{trajectory_id}")

        fig_id = self.path_registry.get_fileid(file_name=path, type=FileType.FIGURE)
        movie_desc = utils.make_movie(matrix, path=path, source_id=trajectory_id)
        self.path_registry.map_path(fig_id, path, movie_desc)
        utils.save_matrix_frame(
            matrix, f"{self.path_registry.ckpt_figures}/contact_{trajectory_id}"
        )
        return "Contact Matrix Figure created with ID: " + fig_id

    def _arun(self):
        pass

    def validate_inputs(self, input):
        input = input.get("action_input", input)
        input = input.get("input", input)
        trajectory_id = input.get("trajectory_fileid", None)
        topology_id = input.get("topology_fileid", None)
        selection = input.get("selection", "all")
        cutoff = input.get("cutoff", 0.8)
        if not trajectory_id:
            raise ValueError("Incorrect Inputs: trajectory_fileid is required")
        if not topology_id:
            raise ValueError("Incorrect Inputs: topology_fileid is required")
        fileids = self.path_registry.list_path_names()
        error = ""
        system_message = ""
        if trajectory_id not in fileids:
            error += " Trajectory File ID not in path registry"
        if topology_id not in fileids:
            error += " Topology File ID not in path registry"
        keys = input.keys()
        for key in keys:
            if key not in [
                "trajectory_fileid",
                "topology_fileid",
                "cutoff",
                "selection",
            ]:
                system_message += f"{key} is not part of admitted tool inputs"
        if error == "":
            error = None
        return {
            "trajectory_fileid": trajectory_id,
            "topology_fileid": topology_id,
            "selection": selection,
            "cutoff": cutoff,
            "error": error,
            "system_message": system_message,
        }


def get_values(self, input):
    traj_id = input.get("trajectory_fileid")
    top_id = input.get("topology_fileid")
    sel = input.get("selection")
    cutoff = input.get("cutoff")
    error = input.get("error")
    syst_mes = input.get("system_message")

    return traj_id, top_id, sel, cutoff, error, syst_mes
