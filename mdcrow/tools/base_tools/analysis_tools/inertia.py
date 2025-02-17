from typing import Optional

import matplotlib.pyplot as plt
import mdtraj as md
import numpy as np
from langchain.tools import BaseTool

from mdcrow.utils import FileType, PathRegistry, load_single_traj, save_to_csv


class MOIFunctions:
    def __init__(self, path_registry, top_fileid, traj_fileid=None, mol_name=None):
        self.path_registry = path_registry
        self.moments_of_inertia = None
        self.min_moi = None
        self.avg_moi = None

        self.mol_name = mol_name if mol_name else top_fileid.replace("top_", "")
        self.traj = load_single_traj(self.path_registry, top_fileid, traj_fileid)

    def calculate_moment_of_inertia(self):
        """
        Calculate principal moments of inertia for a molecule or protein.
        Expected shape of moments_of_inertia: (n_frames, 3)
        """
        print("Calculating moments of inertia...")
        inertia_tensors = md.compute_inertia_tensor(self.traj)
        principal_moments = np.empty((len(inertia_tensors), 3))
        for i, tensor in enumerate(inertia_tensors):
            eigenvalues, _ = np.linalg.eigh(tensor)
            principal_moments[i] = eigenvalues

        self.moments_of_inertia = principal_moments
        self.min_moi = np.min(self.moments_of_inertia)  # min of all frames & moments
        self.avg_moi = np.mean(self.moments_of_inertia)  # avg of all frames & moments

        # save to file
        description = f"Moments of inertia for {self.mol_name}"
        csv_file_id = save_to_csv(
            self.path_registry,
            self.moments_of_inertia,
            f"MOI_{self.mol_name}",
            description,
            header="I1,I2,I3",
        )
        message = (
            f"Data saved with file ID {csv_file_id}. \n"
            f"Average Moment of Inertia of all frames: {self.avg_moi:.2f}. \n"
        )
        return message

    def plot_moi(self):
        """
        Analyze and visualize the principal moments of inertia.
        """
        message = ""
        if self.moments_of_inertia is None:
            message += self.calculate_moment_of_inertia()

        if self.traj.n_frames == 1:  # only one frame
            moi_string = ", ".join(f"{moi:.2f}" for moi in self.moments_of_inertia[0])
            message += (
                "Only one frame in trajectory, no plot generated. \n"
                f"Principal Moments of Inertia: {moi_string}. \n"
            )
            return message

        fig_analysis = f"MOI_{self.mol_name}"
        fig_name = self.path_registry.write_file_name(
            type=FileType.FIGURE,
            fig_analysis=fig_analysis,
            file_format="png",
        )
        fig_id = self.path_registry.get_fileid(file_name=fig_name, type=FileType.FIGURE)

        plt.axhline(self.avg_moi, label="$I_avg$", linestyle="--", color="black")
        plt.axhline(self.min_moi, label="$I_min$", linestyle="--", color="red")
        plt.plot(self.moments_of_inertia[:, 0], label="$I_1$")  # smallest MOI
        plt.plot(self.moments_of_inertia[:, 1], label="$I_2$")
        plt.plot(self.moments_of_inertia[:, 2], label="$I_3$")  # largest MOI
        plt.xlim(0, self.traj.n_frames - 1)
        plt.xlabel("Frame")
        plt.ylabel("Moments of Inertia")
        plt.title("Moments of Inertia over Time")
        plt.legend()

        plt.savefig(f"{self.path_registry.ckpt_figures}/{fig_name}")
        plt.close()
        print(f"Plot of moments of inertia saved to {fig_name}")
        self.path_registry.map_path(
            fig_id,
            f"{self.path_registry.ckpt_figures}/{fig_name}",
            description=f"Plot of moments of inertia over time for {self.mol_name}",
        )
        message += (
            f"Plot of moments of inertia over time saved with plot ID {fig_id}. \n"
        )
        return message


class MomentOfInertia(BaseTool):
    name = "MomentOfInertia"
    description = (
        "Compute the moment of inertia tensors for a molecule or protein."
        "\nInputs: \n"
        "\t(str) File ID for the topology file. \n"
        "\t(str, optional) File ID for the trajectory file. \n"
        "\t(str, optional) Molecule or protein name. \n"
    )
    path_registry: PathRegistry | None

    def __init__(self, path_registry=None):
        super().__init__()
        self.path_registry = path_registry

    def _run(
        self,
        top_fileid: str,
        traj_fileid: Optional[str] = None,
        molecule_name: Optional[str] = None,
    ) -> str:
        try:
            moi_funcs = MOIFunctions(
                self.path_registry, top_fileid, traj_fileid, molecule_name
            )
            msg = moi_funcs.plot_moi()
            return f"Succeeded. {msg}"
        except Exception as e:
            return f"Failed. {type(e).__name__}: {e}"
