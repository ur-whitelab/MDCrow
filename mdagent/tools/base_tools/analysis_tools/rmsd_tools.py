import os
from typing import Optional, Type

import matplotlib.pyplot as plt
import MDAnalysis as mda
import numpy as np
import streamlit as st
from langchain.tools import BaseTool
from MDAnalysis.analysis import align, diffusionmap, rms
from pydantic import BaseModel, Field

from mdagent.utils import FileType, PathRegistry

# all things related to RMSD as 'standard deviation'
# 1  RMSD between two protein conformations or trajectories (1D scalar value)
# 2. time-dependent RMSD of the whole trajectory with all or selected atoms
# 3. pairwise RMSD
# 4. RMSF - root mean square fluctuation


class RMSDFunctions:
    def __init__(self, path_registry, pdb, traj, ref=None, ref_traj=None):
        self.path_registry = path_registry
        self.pdb_file = self.path_registry.get_mapped_path(pdb)
        self.trajectory = self.path_registry.get_mapped_path(traj)
        self.pdb_name = os.path.splitext(os.path.basename(self.pdb_file))[0]
        self.ref_file = self.path_registry.get_mapped_path(ref)
        self.ref_trajectory = self.path_registry.get_mapped_path(ref_traj)
        if self.ref_file:
            self.ref_name = os.path.splitext(os.path.basename(self.ref_file))[0]

    def calculate_rmsd(
        self,
        rmsd_type="rmsd",
        selection="backbone",
        plot=True,
    ):
        if self.trajectory is None or self.pdb_file is None:
            raise FileNotFoundError("PDB and trajectory files are required.")
        self.filename = f"{rmsd_type}_{self.pdb_name}"

        if rmsd_type == "rmsd":
            if self.ref_file:
                print("Calculating 1-D RMSD between two sets of coordinates...")
                st.markdown(
                    "Calculating 1-D RMSD between two sets of coordinates...",
                    unsafe_allow_html=True,
                )
                return self.compute_rmsd_2sets(selection=selection)
            else:
                print("Calculating time-dependent RMSD...")
                st.markdown(
                    "Calculating time-dependent RMSD...", unsafe_allow_html=True
                )
                return self.compute_rmsd(selection=selection, plot=plot)
        elif rmsd_type == "pairwise_rmsd":
            print("Calculating pairwise RMSD...")
            st.markdown("Calculating pairwise RMSD...", unsafe_allow_html=True)
            return self.compute_2d_rmsd(selection=selection, plot_heatmap=plot)
        elif rmsd_type == "rmsf":
            print("Calculating root mean square fluctuation (RMSF)...")
            st.markdown(
                "Calculating root mean square fluctuation (RMSF)...",
                unsafe_allow_html=True,
            )
            return self.compute_rmsf(selection=selection, plot=plot)
        else:
            raise ValueError(
                "Invalid rmsd_type. Please choose from 'rmsd', 'pairwise_rmsd', 'rmsf'"
            )

    def compute_rmsd_2sets(self, selection="backbone"):
        # simple RMSD calculation between two different sets of protein coordinates
        # returns scalar value
        if self.trajectory and self.ref_trajectory:
            u = mda.Universe(self.pdb_file, self.trajectory)
            ref = mda.Universe(self.ref_file, self.ref_trajectory)
        else:
            u = mda.Universe(self.pdb_file)
            ref = mda.Universe(self.ref_file)
        rmsd_value = rms.rmsd(
            u.select_atoms(selection).positions,  # coordinates to align
            ref.select_atoms(selection).positions,  # reference coordinates
            center=True,  # subtract the center of geometry
            superposition=True,
        )  # superimpose coordinates
        return f"{rmsd_value}\n"

    def compute_rmsd(self, selection="backbone", plot=True):
        # 1D time-dependent RMSD, gives one scalar value for each timestep
        if self.trajectory is None:
            raise ValueError("trajectory file is required for time-dependent 1D RMSD")
        u = mda.Universe(self.pdb_file, self.trajectory)
        R = rms.RMSD(u, select=selection)
        R.run()

        # save to file
        np.savetxt(
            f"{self.filename}.csv",
            R.results.rmsd,
            fmt=["%d", "%f", "%f"],
            delimiter=",",
            header="Frame,Time,RMSD",
            comments="",
        )
        avg_rmsd = np.mean(R.results.rmsd[:, 2])  # rmsd values are in 3rd column
        final_rmsd = R.results.rmsd[-1, 2]
        message = f"""Calculated RMSD for each timestep with respect\
        to the initial frame. Saved to {self.filename}.csv. """
        self.path_registry.map_path(
            f"{self.filename}.csv", f"{self.filename}.csv", message
        )
        message += f"Average RMSD is {avg_rmsd} \u212B. "
        message += f"Final RMSD is {final_rmsd} \u212B.\n"

        if plot:
            plt.plot(R.results.rmsd[:, 0], R.results.rmsd[:, 2], label=str(selection))
            plt.xlabel("Frame")
            plt.ylabel("RMSD ($\AA$)")
            plt.title("Time-Dependent RMSD")
            plt.legend()
            plt.show()
            if not os.path.exists("files/figures"):  # PR: Needed to avoid error
                os.makedirs("files/figures")
            plot_name = self.path_registry.write_file_name(
                type=FileType.FIGURE,
                fig_analysis=self.filename,
                file_format="png",
            )
            plot_id = self.path_registry.get_fileid(
                file_name=plot_name, type=FileType.FIGURE
            )
            plt.savefig(f"files/figures/{plot_name}.png")
            plot_message = (
                f"Plotted RMSD over time for{self.pdb_file}."
                f" Saved with plot id {plot_id}.\n"
            )
            message += plot_message
            self.path_registry.map_path(
                plot_id, f"files/figures/{plot_name}", plot_message
            )
        return message

    def compute_2d_rmsd(self, selection="backbone", plot_heatmap=True):
        # pairwise RMSD, also known as 2D RMSD, gives a matrix of RMSD values
        if self.trajectory is None:
            raise ValueError("trajectory file is required for pairwise RMSD")
        u = mda.Universe(self.pdb_file, self.trajectory)
        if self.ref_file and self.ref_trajectory:
            ref = mda.Universe(self.ref_file, self.ref_trajectory)
        else:
            ref = None

        if ref is None:
            # pairwise RMSD of a trajectory to itself
            align.AlignTraj(u, u, select=selection, in_memory=True).run()
            matrix = diffusionmap.DistanceMatrix(u, select=selection).run()
            pairwise_matrix = matrix.results.dist_matrix
            x_label = y_label = "Frame"
        else:
            pairwise_matrix = np.zeros((len(u.trajectory), len(ref.trajectory)))
            for i, frame in enumerate(u.trajectory):
                r = rms.RMSD(ref, u, select=selection, ref_frame=i).run()
                pairwise_matrix[i] = r.results.rmsd[:, 2]
            x_label = f"Frame ({self.ref_name})"
            y_label = f"Frame ({self.pdb_name})"
        np.savetxt(
            f"{self.filename}.csv",
            pairwise_matrix,
            delimiter=",",
        )
        message = f"Saved pairwise RMSD matrix to {self.filename}.csv.\n"
        self.path_registry.map_path(
            f"{self.filename}.csv", f"{self.filename}.csv", message
        )
        if plot_heatmap:
            plt.imshow(pairwise_matrix, cmap="viridis")
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.colorbar(label=r"RMSD ($\AA$)")
            plt.show()
            plt.savefig(f"{self.filename}.png")
            message += f"Plotted pairwise RMSD matrix. Saved to {self.filename}.png.\n"
            self.path_registry.map_path(
                f"{self.filename}.png", f"{self.filename}.png", message
            )
        return message

    def compute_rmsf(self, selection="backbone", plot=True):
        # calculate RMSF (root mean square fluctuation)
        u = mda.Universe(self.pdb_file, self.trajectory)

        # use averages as a reference for aligning
        average = align.AverageStructure(u, u, select=selection, ref_frame=0).run()
        align_ref = average.results.universe
        align.AlignTraj(u, align_ref, select=selection, in_memory=True).run()

        # Compute RMSF
        atoms = u.select_atoms(selection)
        R = rms.RMSF(atoms).run()
        rmsf = R.results.rmsf

        # Save to a text file
        rmsf_data = np.column_stack((atoms.resids, rmsf))
        np.savetxt(
            f"{self.filename}.csv",
            rmsf_data,
            delimiter=",",
            header="Residue_ID,RMSF",
            comments="",
        )
        message = f"Saved RMSF data to {self.filename}.csv.\n"
        self.path_registry.map_path(
            f"{self.filename}.csv", f"{self.filename}.csv", message
        )

        # Plot RMSF
        if plot:
            plt.figure(figsize=(5, 3))
            plt.plot(atoms.resnums, rmsf, label=str(selection))
            plt.xlabel("Residue Number")
            plt.ylabel("RMSF ($\AA$)")
            plt.title("Root Mean Square Fluctuation")
            plt.legend()
            plt.show()
            plt.savefig(f"{self.filename}.png")
            message += f"Plotted RMSF. Saved to {self.filename}.png.\n"
            self.path_registry.map_path(
                f"{self.filename}.png", f"{self.filename}.png", message
            )
        return message


class RMSDInputSchema(BaseModel):
    rmsd_type: str = Field(
        description="""type of RMSD calculation
        to perform. Choose from 'rmsd', 'pairwise_rmsd', 'rmsf'.
        'rmsd': any 1-D root mean square deviation calculations.
        'pairwise_rmsd': 2D root mean square deviation calculation.
            pairwise RMSD matrix is computed. Either
            trajectory against itself or a given reference.
        'rmsf': root mean square fluctuation. it computes the average
            fluctuation for each residue for the entire trajectory.
        """
    )
    pdb_file: str = Field(
        description="file with .pdb extension contain protein of interest"
    )
    trajectory: Optional[str] = Field(
        description="trajectory file for protein of interest"
    )
    ref_file: Optional[str] = Field(
        description="file with .pdb extension used as reference"
    )
    ref_trajectory: Optional[str] = Field(
        description="trajectory file used as reference"
    )
    selection: Optional[str] = Field(
        description="""selected atoms using MDAnalysis selection syntax."""
    )
    plot: Optional[bool] = Field(
        description="""Only use it to set False
        to disable making plots if prompted."""
    )


class RMSDCalculator(BaseTool):
    name: str = "RMSDCalculator"
    description: str = """Useful for calculating RMSD from output files
    such as PDB, PSF, DCD, etc. Types of RMSD this tool can do:
    1. 1-D root mean square deviation (RMSD)
    2. 2-D or pairwise root mean square deviation (RMSD) matrix
    3. root mean square fluctuation (RMSF)
    Make sure to provide any necessary files for a chosen RMSD type."""
    args_schema: Type[BaseModel] = RMSDInputSchema
    path_registry: Optional[PathRegistry]

    def __init__(self, path_registry: Optional[PathRegistry] = None):
        super().__init__()
        self.path_registry = path_registry

    def _run(
        self,
        rmsd_type: str,
        pdb_file: str,
        trajectory: Optional[str] = None,
        ref_file: Optional[str] = None,
        ref_trajectory: Optional[str] = None,
        selection: str = "backbone",
        plot: bool = True,
    ):
        try:
            rmsd = RMSDFunctions(
                self.path_registry, pdb_file, trajectory, ref_file, ref_trajectory
            )
            message = rmsd.calculate_rmsd(rmsd_type, selection, plot)
        except ValueError as e:
            return (
                f"ValueError: {e}. \nMake sure to provide valid PBD "
                "file and binding site using MDAnalysis selection syntax."
            )
        except FileNotFoundError as e:
            return str(e)
        except Exception as e:
            return f"Something went wrong. {type(e).__name__}: {e}"
        return message

    def _arun(self, **query):
        """Use the tool asynchronously."""
        raise NotImplementedError("This tool does not support async")
