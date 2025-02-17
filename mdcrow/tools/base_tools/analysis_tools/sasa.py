from typing import Optional

import matplotlib.pyplot as plt
import mdtraj as md
import numpy as np
from langchain.tools import BaseTool

from mdcrow.utils import FileType, PathRegistry, load_single_traj, save_to_csv


class SASAFunctions:
    def __init__(self, path_registry, top_fileid, traj_fileid=None, mol_name=None):
        """
        Initialize the SASAFunctions class with topology and/or trajectory files.

        Parameters:
        path_registry (PathRegistry): mapping file IDs to file paths.
        top_fileid (str): File ID for the topology file.
        traj_fileid (str, optional): File ID for the trajectory file.
        mol_name (str, optional): Name of the molecule or protein.
        """
        self.path_registry = path_registry
        self.residue_sasa = None
        self.total_sasa = None

        print("Loading trajectory ...")
        self.molecule_name = mol_name if mol_name else top_fileid.replace("top_", "")
        self.traj = load_single_traj(self.path_registry, top_fileid, traj_fileid)

    def calculate_sasa(self, probe_radius=0.14):
        """
        Calculate the Solvent Accessible Surface Area (SASA) for each
        frame in the trajectory using Shrake-Rupley algorithm.

        Parameters:
        probe_radius (float, optional): The radius of the probe used to calculate SASA.
            Default is 0.14 nm (1.4 Å).
        """
        print("Calcuating SASA ...")
        self.residue_sasa = md.shrake_rupley(
            self.traj, probe_radius=probe_radius, mode="residue"
        )
        self.total_sasa = self.residue_sasa.sum(axis=1)

        # save total SASA to file --> can use for autocorrelation analysis
        description = f"Total SASA values for {self.molecule_name}"
        csv_header = "Total SASA (nm²)"
        csv_file_id = save_to_csv(
            self.path_registry,
            self.total_sasa,
            f"sasa_{self.molecule_name}",
            description,
            csv_header,
        )
        # TODO: also save per-residue or per-atom SASA (longer computation time)?

        return f"SASA values computed and saved with File ID {csv_file_id}. "

    def plot_sasa(self):
        """
        Plot the total SASA and per-residue SASA over time.
        """
        message = ""
        if self.total_sasa is None or self.residue_sasa is None:
            message += self.calculate_sasa()

        # if there's only one frame, don't plot
        if self.traj.n_frames == 1:
            message += (
                " Only one frame in trajectory. No SASA plot generated. "
                f"Total Available Surface Area is {self.total_sasa[0]:,.2f} nm²."
            )
            return message

        fig_analysis = f"sasa_{self.molecule_name}"
        fig_name = self.path_registry.write_file_name(
            type=FileType.FIGURE, fig_analysis=fig_analysis, file_format="png"
        )
        fig_id = self.path_registry.get_fileid(file_name=fig_name, type=FileType.FIGURE)
        plt.figure(figsize=(10, 5))
        plt.subplot(121)
        plt.plot(self.total_sasa)
        plt.xlabel("Frame")
        plt.ylabel("Total SASA (nm²)")
        plt.title("Total SASA over Time")

        # average SASA per residue
        if self.traj.n_residues > 1:
            plt.subplot(122)
            avg_residue_sasa = np.mean(self.residue_sasa, axis=0)
            plt.plot(avg_residue_sasa)
            plt.xlabel("Residue")
            plt.ylabel("Average Area (nm²)")
            plt.title("Average SASA per Residue")
        plt.tight_layout()
        plt.savefig(
            f"{self.path_registry.ckpt_figures}/{fig_name}", bbox_inches="tight"
        )
        plt.close()
        print(f"SASA plot saved to {fig_name}")
        self.path_registry.map_path(
            fig_id,
            f"{self.path_registry.ckpt_figures}/{fig_name}",
            description=f"Plot of SASA over time for {self.molecule_name}",
        )
        message += f"SASA analysis completed. Saved the plot with plot ID {fig_id}. "
        return message


class SolventAccessibleSurfaceArea(BaseTool):
    name = "SolventAccessibleSurfaceArea"
    description = (
        "Compute the Solvent Accessible Surface Area (SASA) for a molecule or protein."
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
            sasa_analysis = SASAFunctions(
                self.path_registry, top_fileid, traj_fileid, molecule_name
            )
            return f"Succeeded. {sasa_analysis.plot_sasa()}"
        except Exception as e:
            return f"Failed. {type(e).__name__}: {e}"
