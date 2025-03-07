from typing import List, Optional

import matplotlib.pyplot as plt
import mdtraj as md
from state import MDCrowState
from utils import FileType


def validate_input(path_registry, input):
    input = input.get("input", input)

    input = input.get("action_input", input)

    trajectory_id = input.get("trajectory_fileid", None)

    topology_id = input.get("topology_fileid", None)

    stride = input.get("stride", None)

    atom_indices = input.get("atom_indices", None)

    if not trajectory_id:
        raise ValueError(
            "Incorrect Inputs: Trajectory file ID ('trajectory_fileid')is required"
        )

    # check if trajectory id is valid
    fileids = path_registry.list_path_names()

    if trajectory_id not in fileids:
        raise ValueError("Trajectory File ID not in path registry")

    path_to_traj = path_registry.get_mapped_path(trajectory_id)

    ending = path_to_traj.split(".")[-1]
    if ending in ["dcd", "xtc", "xyz"]:
        # requires topology
        if not topology_id:
            raise ValueError(
                "Incorrect Inputs: "
                "Topology file (topology_fileid) is required for trajectory "
                "file with extension {}".format(ending)
            )
        if topology_id not in fileids:
            raise ValueError("Topology File ID not in path registry")

    elif ending in ["hdf5", "h5", "pdb"]:
        # does not require topology
        pass

    else:
        raise ValueError(
            "Invalid file extension for trajectory file. "
            "For the moment only supported extensions are: "
            "dcd, xtc, hdf5, h5, xyz, pdb"
        )

    if stride:
        if not isinstance(stride, int):
            try:
                stride = int(stride)
                if stride <= 0:
                    raise ValueError(
                        "Incorrect Inputs: "
                        "Stride must be a positive integer "
                        "or None for default value of 1."
                    )
            except ValueError:
                raise ValueError(
                    "Incorrect Inputs: Stride must be an integer "
                    "or None for default value of 1."
                )
        else:
            if stride <= 0:
                raise ValueError("Incorrect Inputs: Stride must be a positive integer.")

    if atom_indices:
        try:
            atom_indices = list(map(int, atom_indices.split(",")))
        except ValueError:
            raise ValueError(
                "Incorrect Inputs: Atom indices must be a comma "
                "separated list of integers or None for all atoms."
            )
    inputs = {
        "trajectory_fileid": trajectory_id,
        "topology_fileid": topology_id,
        "stride": stride,
        "atom_indices": atom_indices,
    }

    return inputs


# TODO: Add pairs of atoms to calculate RDF within the tool
# pairs: Optional[str] = Field(None, description="Pairs of atoms to calculate RDF ")
async def compute_rdf(
    state: MDCrowState,
    trajectory_fileid: str,
    topology_fileid: Optional[str] = None,
    stride: Optional[int] = None,
    atom_indices: Optional[List[int]] = None,
):
    """

        Calculate the radial distribution function (RDF) of a trajectory \
        of a protein with respect to water molecules using the trajectory file ID \
        (trajectory_fileid) and optionally the topology file ID (topology_fileid).

    Args:
        trajectory_fileid (str): File ID for the trajectory file.
            Supported formats: dcd, hdf5, xtc, xyz.
        topology_fileid (Optional[str], optional): File ID for the topology file.
        stride (Optional[int], optional): Stride value for reading the trajectory.
            Determines how frequently frames are sampled.
        atom_indices (Optional[List[int]], optional): List of atom indices to load
            in the trajectory.

    """
    inputs = {
        "trajectory_fileid": trajectory_fileid,
        "topology_fileid": topology_fileid,
        "stride": stride,
        "atom_indices": atom_indices,
    }
    try:
        inputs = validate_input(state.path_registry, inputs)
    except ValueError as e:
        if "Incorrect Inputs" in str(e):
            print("Error in Inputs in RDF tool: ", str(e))
            return ("Failed. Error in Inputs", str(e)), 0, False
        elif "Invalid file extension" in str(e):
            print("File Extension Not Supported in RDF tool: ", str(e))
            return ("Failed. File Extension Not Supported", str(e)), 0, False
        elif "not in path registry" in str(e):
            print("File ID not in Path Registry in RDF tool: ", str(e))
            return ("Failed. File ID not in Path Registry", str(e)), 0, False
        else:
            raise ValueError(f"Error during inputs in RDF tool {e}")

    trajectory_id = inputs["trajectory_fileid"]
    topology_id = inputs["topology_fileid"]
    # stride = inputs["stride"]  # ignore
    stride = int(inputs["stride"]) if isinstance(inputs["stride"], (str, int)) else None
    ## Pre-commit was mad at this assignment so had to make sure is actually a list(int)
    ## or None [JM].
    if isinstance(inputs["atom_indices"], int):
        atom_indices = [inputs["atom_indices"]]  # Convert single int to list
    elif isinstance(inputs["atom_indices"], str):
        atom_indices = [
            int(x) for x in inputs["atom_indices"].split(",")
        ]  # Convert comma-separated string to list
    elif isinstance(inputs["atom_indices"], list):
        atom_indices = inputs["atom_indices"]
    else:
        atom_indices = None  # Keep None if not provided

    path_to_traj = state.path_registry.get_mapped_path(trajectory_id)
    ending = path_to_traj.split(".")[-1]
    if ending in ["dcd", "xtc", "xyz"]:
        path_to_top = state.path_registry.get_mapped_path(topology_id)
        traj = md.load(
            path_to_traj, top=path_to_top, stride=stride, atom_indices=atom_indices
        )
    else:
        # hdf5, h5, pdb already checked in validation of inputs
        traj = md.load(path_to_traj, stride=stride, atom_indices=atom_indices)
    try:
        r, gr = md.compute_rdf(
            traj,
            traj.topology.select_pairs(
                ("protein and backbone and " "(name C or name N or name CA)"),
                "water and name O",
            ),
            r_range=(0.1, 2),  # Adjust these values based on your system
            bin_width=0.005,
        )
    except Exception as e:
        # not sure what exceptions to catch for now, will handle them as they come
        print("Error in RDF calculation:", str(e))
        raise Exception("Failed. Error in RDF calculation: ", str(e))
    # save plot
    plot_name_save = f"{state.path_registry.ckpt_figures}/rdf_{trajectory_id}.png"
    try:
        fig, ax = plt.subplots()
        ax.plot(r, gr)
        ax.set_xlabel(r"$r$ (nm)")
        ax.set_ylabel(r"$g(r)$")
        ax.set_title("RDF")
        plt.savefig(plot_name_save)
        plot_name = state.path_registry.write_file_name(
            type=FileType.FIGURE,
            fig_analysis="rdf",
            file_format="png",
            Log_id=trajectory_id,
        )
        fig_id = state.path_registry.get_fileid(plot_name, type=FileType.FIGURE)
        file_path = f"{state.path_registry.ckpt_figures}/rdf_{trajectory_id}.png"
        plt.savefig(file_path)
        state.path_registry.map_path(
            fig_id,
            file_path,
            description=f"RDF plot for the trajectory file with id: {trajectory_id}",
        )
        plt.close()
        return f"Succeeded. RDF calculated. Analysis plot: {fig_id}", 0, False
    except Exception as e:
        return f"Failed. Couldnt save rdf plot {type(e).__name__}: {e}", 0, False


# class RDFTool(BaseTool):
#     name = "RDFTool"
#     description = (
#         "Calculate the radial distribution function (RDF) of a trajectory "
#         "of a protein with respect to water molecules using the trajectory file ID "
#         "(trajectory_fileid) and optionally the topology file ID (topology_fileid). "
#     )
#     args_schema = RDFToolInput
#     path_registry: Optional[PathRegistry]

#     def __init__(self, path_registry: Optional[PathRegistry] = None):
#         super().__init__()
#         self.path_registry = path_registry

#     def _run(self, **input):
#         try:
#             inputs = self.validate_input(input)
#         except ValueError as e:
#             if "Incorrect Inputs" in str(e):
#                 print("Error in Inputs in RDF tool: ", str(e))
#                 return ("Failed. Error in Inputs", str(e))
#             elif "Invalid file extension" in str(e):
#                 print("File Extension Not Supported in RDF tool: ", str(e))
#                 return ("Failed. File Extension Not Supported", str(e))
#             elif "not in path registry" in str(e):
#                 print("File ID not in Path Registry in RDF tool: ", str(e))
#                 return ("Failed. File ID not in Path Registry", str(e))
#             else:
#                 raise ValueError(f"Error during inputs in RDF tool {e}")

#         trajectory_id = inputs["trajectory_fileid"]
#         topology_id = inputs["topology_fileid"]
#         stride = inputs["stride"]
#         atom_indices = inputs["atom_indices"]

#         path_to_traj = self.path_registry.get_mapped_path(trajectory_id)
#         ending = path_to_traj.split(".")[-1]
#         if ending in ["dcd", "xtc", "xyz"]:
#             path_to_top = self.path_registry.get_mapped_path(topology_id)
#             traj = md.load(
#                 path_to_traj, top=path_to_top, stride=stride, atom_indices=atom_indices
#             )
#         else:
#             # hdf5, h5, pdb already checked in validation of inputs
#             traj = md.load(path_to_traj, stride=stride, atom_indices=atom_indices)
#         try:
#             r, gr = md.compute_rdf(
#                 traj,
#                 traj.topology.select_pairs(
#                     ("protein and backbone and " "(name C or name N or name CA)"),
#                     "water and name O",
#                 ),
#                 r_range=(0.1, 2),  # Adjust these values based on your system
#                 bin_width=0.005,
#             )
#         except Exception as e:
#             # not sure what exceptions to catch for now, will handle them as they come
#             print("Error in RDF calculation:", str(e))
#             raise ("Failed. Error in RDF calculation: ", str(e))
#         # save plot
#         plot_name_save = f"{self.path_registry.ckpt_figures}/rdf_{trajectory_id}.png"
#         fig, ax = plt.subplots()
#         ax.plot(r, gr)
#         ax.set_xlabel(r"$r$ (nm)")
#         ax.set_ylabel(r"$g(r)$")
#         ax.set_title("RDF")
#         plt.savefig(plot_name_save)
#         plot_name = self.path_registry.write_file_name(
#             type=FileType.FIGURE,
#             fig_analysis="rdf",
#             file_format="png",
#             Log_id=trajectory_id,
#         )
#         fig_id = self.path_registry.get_fileid(plot_name, type=FileType.FIGURE)
#         file_path = f"{self.path_registry.ckpt_figures}/rdf_{trajectory_id}.png"
#         plt.savefig(file_path)
#         self.path_registry.map_path(
#             fig_id,
#             file_path,
#             description=f"RDF plot for the trajectory file with id: {trajectory_id}",
#         )
#         plt.close()
#         return f"Succeeded. RDF calculated. Analysis plot: {fig_id}"

#     def _arun(self, input):
#         pass

#     def validate_input( input):
#         input = input.get("input", input)

#         input = input.get("action_input", input)

#         trajectory_id = input.get("trajectory_fileid", None)

#         topology_id = input.get("topology_fileid", None)

#         stride = input.get("stride", None)

#         atom_indices = input.get("atom_indices", None)

#         if not trajectory_id:
#             raise ValueError(
#                 "Incorrect Inputs: Trajectory file ID ('trajectory_fileid')is required"
#             )

#         # check if trajectory id is valid
#         fileids = self.path_registry.list_path_names()

#         if trajectory_id not in fileids:
#             raise ValueError("Trajectory File ID not in path registry")

#         path_to_traj = self.path_registry.get_mapped_path(trajectory_id)

#         ending = path_to_traj.split(".")[-1]
#         if ending in ["dcd", "xtc", "xyz"]:
#             # requires topology
#             if not topology_id:
#                 raise ValueError(
#                     "Incorrect Inputs: "
#                     "Topology file (topology_fileid) is required for trajectory "
#                     "file with extension {}".format(ending)
#                 )
#             if topology_id not in fileids:
#                 raise ValueError("Topology File ID not in path registry")

#         elif ending in ["hdf5", "h5", "pdb"]:
#             # does not require topology
#             pass

#         else:
#             raise ValueError(
#                 "Invalid file extension for trajectory file. "
#                 "For the moment only supported extensions are: "
#                 "dcd, xtc, hdf5, h5, xyz, pdb"
#             )

#         if stride:
#             if not isinstance(stride, int):
#                 try:
#                     stride = int(stride)
#                     if stride <= 0:
#                         raise ValueError(
#                             "Incorrect Inputs: "
#                             "Stride must be a positive integer "
#                             "or None for default value of 1."
#                         )
#                 except ValueError:
#                     raise ValueError(
#                         "Incorrect Inputs: Stride must be an integer "
#                         "or None for default value of 1."
#                     )
#             else:
#                 if stride <= 0:
#                     raise ValueError(
#                         "Incorrect Inputs: " "Stride must be a positive integer."
#                     )

#         if atom_indices:
#             try:
#                 atom_indices = list(map(int, atom_indices.split(",")))
#             except ValueError:
#                 raise ValueError(
#                     "Incorrect Inputs: Atom indices must be a comma "
#                     "separated list of integers or None for all atoms."
#                 )
#         inputs = {
#             "trajectory_fileid": trajectory_id,
#             "topology_fileid": topology_id,
#             "stride": stride,
#             "atom_indices": atom_indices,
#         }

#         return inputs
