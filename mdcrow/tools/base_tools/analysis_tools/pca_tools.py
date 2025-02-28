import warnings
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from sklearn.decomposition import PCA

from mdcrow.utils import FileType, PathRegistry, load_single_traj


class PCA_analysis:
    def __init__(
        self,
        path_registry,
        pc_percentage,
        traj,
        sim_id,
        selection="backbone",
    ):
        self.path_registry = path_registry
        self.traj = traj
        self.sim_id = sim_id
        if pc_percentage > 1:
            pc_percentage /= 100
        self.pc_percentage = pc_percentage
        self.atom_indices = self.traj.top.select(selection)
        self.selection = selection
        self.pc = PCA(n_components=30)
        self.reduce_dim = None
        self._sub_pcs = None
        self.n_pcs = None

    def _align_trajectory(self):
        ###########MD TRAJ ALIGNMENT################
        try:
            self.traj.superpose(self.traj, frame=0, atom_indices=self.atom_indices)
            return "Trajectory aligned to the first frame. "
        except Exception as e:
            warnings.warn(f"Error aligning trajectory: {str(e)}", RuntimeWarning)
            return "Trajectory not aligned. "

    def get_pc(self):
        align_message = self._align_trajectory()
        self.reduce_dim = self.pc.fit_transform(
            self.traj.xyz[:, self.atom_indices].reshape(
                self.traj.n_frames, len(self.atom_indices) * 3
            )
        )
        if "not aligned" in align_message:
            warnings.warn(
                "The trajectory is not aligned, but the PCA analysis"
                "will continue. Be wary of the final results",
                RuntimeWarning,
            )
        return "PCA done" + align_message

    def _sub_array_sum_to_m(self, arr, M):
        result = []
        sum = 0
        for value in arr:
            if sum >= M:
                break
            result.append(value)
            sum += value
        return result

    def _get_number_pcs(self):
        self._sub_pcs = self._sub_array_sum_to_m(
            self.pc.explained_variance_ratio_, self.pc_percentage
        )
        self.n_pcs = len(self._sub_pcs)
        if self.n_pcs > 3:
            self.n_pcs = 3
        return self.n_pcs

    def _make_transformation(self):
        self.transformed = self.reduce_dim[:, : self.n_pcs]

    def _make_df(self):
        self._get_number_pcs()
        self._make_transformation()
        self.pc_df = pd.DataFrame(
            self.transformed, columns=["PC{}".format(i + 1) for i in range(self.n_pcs)]
        )
        self.pc_df["Time (ps)"] = self.pc_df.index * self.traj.timestep

    def _cosine_content(self, pca_space, i):
        """Measure the cosine content of the PCA projection.

        The cosine content of pca projections can be used as an indicator if a
        simulation is converged. Values close to 1 are an indicator that the
        simulation isn't converged. For values below 0.7 no statement can be made.
        If you use this function please cite [BerkHess1]_.
        References
        ----------
        .. [BerkHess1] Berk Hess. Convergence of sampling in protein simulations.
                    Phys. Rev. E 65, 031910 (2002).
        """
        t = np.arange(len(pca_space))
        T = len(pca_space)
        cos = np.cos(np.pi * t * (i + 1) / T)
        return (
            (2.0 / T)
            * (scipy.integrate.simpson(cos * pca_space[:, i])) ** 2
            / scipy.integrate.simpson(pca_space[:, i] ** 2)
        )

    def make_scree_plot(self):
        extra_mess = ""
        if not self.pc:
            extra_mess += self.get_pc()
        if not self.n_pcs:
            self._get_number_pcs()
        cumulative_variance = self.pc.explained_variance_ratio_.cumsum()
        plt.plot(cumulative_variance)
        # Calculate the index where cumulative variance exceeds or meets 95%
        threshold_index = len(self._sub_pcs) - 1

        # Add a horizontal dashed line at 95% threshold
        plt.axvline(
            x=threshold_index,
            color="r",
            linestyle="--",
            label=f"{self.pc_percentage*100}% Contribution at PC {threshold_index+1}",
        )

        plt.xlabel("Principal component")
        plt.ylabel("EigenValue Proportion")
        plt.legend()

        desc = f"Scree Plot of the eigenvalues of simulation {self.sim_id}"
        file_name = self.path_registry.write_file_name(
            FileType.FIGURE,
            fig_analysis="scree_plot",
            file_format="png",
            Sim_id=self.sim_id,
        )
        plot_id = self.path_registry.get_fileid(file_name, FileType.FIGURE)
        path = self.path_registry.ckpt_dir + "/figures/"
        plt.savefig(path + file_name)
        self.path_registry.map_path(plot_id, path + file_name, description=desc)
        return f"Scree Plot saved as {plot_id} ID\n" + extra_mess

    def make_pc_plots(self):
        self._make_df()
        g = sns.PairGrid(
            self.pc_df,
            hue="Time (ps)",
            palette=sns.color_palette("Oranges_d", n_colors=len(self.pc_df)),
        )
        g.map(plt.scatter, marker=".")
        desc = f"PCA Plot comparing the top {self.n_pcs} principal components"
        file_name = self.path_registry.write_file_name(
            FileType.FIGURE, fig_analysis="pca", file_format="png", Sim_id=self.sim_id
        )
        plot_id = self.path_registry.get_fileid(file_name, FileType.FIGURE)
        path = self.path_registry.ckpt_dir + "/figures/"
        plt.savefig(path + file_name)
        self.path_registry.map_path(plot_id, path + file_name, description=desc)
        return f"PCA plots saved as {plot_id} "

    def measure_cosine_convergence(self):
        if not self.pc:
            pc_mess = self.get_pc()
        else:
            pc_mess = ""
        if not self.n_pcs:
            self._get_number_pcs()
            self._make_transformation()

        pc_messages = []
        for i in range(self.n_pcs):
            cc = self._cosine_content(self.transformed, i)
            pc_messages.append(f"Cosine Content for PC {i+1}-{cc:.3f}\n")
        cc_message = f"Cosine Content of each PC: {','.join(pc_messages)}"
        return pc_mess + cc_message

    def run_all(self):
        self.get_pc()
        try:
            scree_plot = self.make_scree_plot()
        except Exception as e:
            raise Exception(f"Error during Scree Plot: str({e})")
        try:
            pc_plots = self.make_pc_plots()
        except Exception as e:
            raise Exception(f"Error during pc plots: str({e})")
        try:
            cos_conv = self.measure_cosine_convergence()
        except Exception as e:
            raise Exception(f"Error during cosine convergence: str({e})")
        return f"Analyses done: {scree_plot}, {pc_plots}, {cos_conv}"


class PCASchema(BaseModel):
    trajectory_fileid: str = Field(
        description="Trajectory File ID of the simulation to be analyzed"
    )
    topology_fileid: str = Field(
        description=("Topology File ID of " "the simulation to be analyzed")
    )
    pc_percentage: Optional[float] = Field(
        95.0, description="Max cumulative percentage of components for analysis"
    )
    analysis: str = Field(
        "all",
        description=(
            "Type of analysis to be done. Availables are: "
            "scree_plot (Saves a scree plot of the eigenvalues), "
            "pca_analysis (gets principal components and saves"
            " a grid plot for visualization) "
            "Cosine (measures the cosine convergence of the top 3 PCs)"
            "all (makes all of the previous analysis)"
        ),
    )
    selection: Optional[str] = Field(
        "backbone and name CA",
        description=(
            "Which selection of atoms from the simulation "
            "to use for the pca analysis"
        ),
    )


class PCATool(BaseTool):
    name = "PCATool"
    description = (
        "Calculate the Principal Analysis Components of a MD trajectory and "
        "performs analysis with them"
    )
    args_schema = PCASchema
    path_registry: Optional[PathRegistry]

    def __init__(self, path_registry: Optional[PathRegistry] = None):
        super().__init__()
        self.path_registry = path_registry

    def _run(self, input):
        try:
            input = self.validate_input(**input)

        except ValueError as e:
            return f"Failed. Error using the PCA Tool: {str(e)}"

        (
            traj_id,
            top_id,
            pc_percentage,
            analysis,
            selection,
            error,
            system_input_message,
        ) = self.get_values(input)

        if error:
            return f"Failed. Error with the tool inputs: {error} "
        if system_input_message == "Tool Messages:":
            system_input_message = ""

        try:
            traj = load_single_traj(
                self.path_registry,
                top_id,
                traj_fileid=traj_id,
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
        return self.run_pca_analysis(
            traj,
            pc_percentage,
            analysis,
            traj_id,
            selection,
            system_input_message,
        )

    def validate_input(self, **input):
        input = input.get("action_input", input)
        input = input.get("input", input)
        trajectory_id = input.get("trajectory_fileid", None)
        topology_id = input.get("topology_fileid", None)
        pc_percentage = input.get("pc_percentage", 95.0)
        analysis = input.get("analysis", "all")
        selection = input.get("selection", "name CA")
        if not trajectory_id:
            raise ValueError("Incorrect Inputs: trajectory_fileid is required")
        if not topology_id:
            raise ValueError("Incorrect Inputs: topology_fileid is required")
        # check if trajectory id is valid
        fileids = self.path_registry.list_path_names()
        error = ""
        system_message = "Tool Messages:"
        if trajectory_id not in fileids:
            error += " Trajectory File ID not in path registry"
        if topology_id not in fileids:
            error += " Topology File ID not in path registry"
        try:
            pc_percentage = float(pc_percentage)
        except ValueError as e:
            if "%" in str(e):
                pc_percentage.replace("%", "")
                try:
                    pc_percentage = float(pc_percentage)
                except Exception as e:
                    error += " pc_percentage value must be a float"
            else:
                error += " pc_percentage value must be a float"
        except Exception:
            error += " pc_percentage value must be a float"

        if analysis.lower() not in [
            "all",
            "pca_analysis",
            "scree_plot",
            "cosine_convergence",
        ]:
            analysis = "all"
            system_message += (
                " analysis arg not recognized, using analysis = 'all' as default"
            )

        if selection not in [
            "backbone",
            "name CA",
            "backbone and name CA",
            "protein",
            "all",
        ]:
            selection = "name CA"  # just alpha carbons
        # get all the kwargs:
        keys = input.keys()
        for key in keys:
            if key not in [
                "trajectory_fileid",
                "topology_fileid",
                "pc_percentage",
                "analysis",
                "selection",
            ]:
                system_message += f"{key} is not part of admitted tool inputs"
        if error == "":
            error = None
        return {
            "trajectory_fileid": trajectory_id,
            "topology_fileid": topology_id,
            "pc_percentage": pc_percentage,
            "analysis": analysis,
            "selection": selection,
            "error": error,
            "system_message": system_message,
        }

    def get_values(self, input):
        traj_id = input.get("trajectory_fileid")
        top_id = input.get("topology_fileid")
        pc_perc = input.get("pc_percentage")
        analysis = input.get("analysis")
        sel = input.get("selection")
        error = input.get("error")
        syst_mes = input.get("system_message")

        return traj_id, top_id, pc_perc, analysis, sel, error, syst_mes

    def run_pca_analysis(
        self,
        traj,
        pc_percentage,
        analysis,
        traj_id,
        selection,
        system_input_message,
    ):
        PCA_container = PCA_analysis(
            self.path_registry,
            pc_percentage=pc_percentage,
            traj=traj,
            sim_id=traj_id,
            selection=selection,
        )
        try:
            if analysis == "all":
                result = PCA_container.run_all()
                return f"Succeeded. {result}. \n\n {system_input_message}"
            if analysis.lower() == "cosine_convergence":
                result = PCA_container.measure_cosine_convergence()
                return (
                    f"Succeeded. Analyses done: {result}. \n\n {system_input_message}"
                )
            if analysis.lower() == "scree_plot":
                result = PCA_container.make_scree_plot()
                return (
                    f"Succeeded. Analyses done: {result}. \n\n {system_input_message}"
                )
        except Exception as e:
            raise (Exception(f"Failed. Error during PCA Tool usage: {str(e)}"))
