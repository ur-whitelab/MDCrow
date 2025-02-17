import matplotlib.pyplot as plt
import mdtraj as md
from langchain.tools import BaseTool

from mdcrow.utils import FileType, PathRegistry, load_single_traj


class HydrogenBondTool(BaseTool):
    """Note that this tool only usees the Baker-Hubbard method for identifying hydrogen bonds.
    Other methods (kabsch-sander, wernet-nilsson) can be implemented later, if desired.
    """

    name = "hydrogen_bond_tool"
    description = (
        "Identifies hydrogen bonds and plots the results from the"
        "provided trajectory data."
        "Input the File ID for the trajectory file and optionally the topology file. "
        "The tool will output the file ID of the results and plot."
    )

    path_registry: PathRegistry | None = None
    freq: float = 0.3

    def __init__(self, path_registry, freq=0.1):
        super().__init__()
        self.path_registry = path_registry
        self.freq = freq

    def compute_hbonds_traj(self, traj):
        hbond_counts = []
        for frame in range(traj.n_frames):
            hbonds = md.baker_hubbard(traj[frame], freq=self.freq)
            hbond_counts.append(len(hbonds))
        return hbond_counts

    def write_hbond_counts_to_file(self, hbond_counts, traj_id):
        output_file = f"{traj_id}_hbond_counts"

        file_name = self.path_registry.write_file_name(
            type=FileType.RECORD, fig_analysis=output_file, file_format="csv"
        )
        file_id = self.path_registry.get_fileid(
            file_name=file_name, type=FileType.FIGURE
        )

        file_path = f"{self.path_registry.ckpt_records}/{file_name}"
        file_path = file_path if file_path.endswith(".csv") else file_path + ".csv"

        with open(file_path, "w") as f:
            f.write("Frame,Hydrogen Bonds\n")
            for frame, count in enumerate(hbond_counts):
                f.write(f"{frame},{count}\n")
        self.path_registry.map_path(
            file_id,
            file_path,
            description=f"Hydrogen bond counts for {traj_id}",
        )
        return f"Data saved to: {file_id}, full path: {file_path}"

    def plot_hbonds_over_time(self, hbond_counts, traj, traj_id):
        fig_analysis = f"hbonds_over_time_{traj_id}"
        plot_name = self.path_registry.write_file_name(
            type=FileType.FIGURE, fig_analysis=fig_analysis, file_format="png"
        )
        plot_id = self.path_registry.get_fileid(
            file_name=plot_name, type=FileType.FIGURE
        )
        plot_path = f"{self.path_registry.ckpt_figures}/{plot_name}"
        plot_path = plot_path if plot_path.endswith(".png") else plot_path + ".png"
        plt.plot(range(traj.n_frames), hbond_counts, marker="o")
        plt.xlabel("Frame")
        plt.ylabel("Number of Hydrogen Bonds")
        plt.title(f"Hydrogen Bonds Over Time for traj {traj_id}")
        plt.grid(True)
        plt.savefig(f"{plot_path}")

        self.path_registry.map_path(
            plot_id,
            plot_path,
            description=f"Plot of hydrogen bonds over time for {traj_id}",
        )
        plt.close()
        plt.clf()
        return f"plot saved to: {plot_id}, full path: {plot_path}"

    def _run(
        self,
        top_file: str,
        traj_file: str | None = None,
    ) -> str:
        try:
            traj_file = (
                traj_file
                if (traj_file is not None) and (traj_file != top_file)
                else None
            )
            traj = load_single_traj(
                path_registry=self.path_registry,
                top_fileid=top_file,
                traj_fileid=traj_file,
                traj_required=False,
            )
            if not traj:
                raise ValueError("Trajectory could not be loaded.")
        except Exception as e:
            return f"Error loading traj: {e}"

        try:
            hbond_counts = self.compute_hbonds_traj(traj)
            rtrn_msg = ""
            if all(count == 0 for count in hbond_counts):
                rtrn_msg += (
                    "No hydrogen bonds found in the trajectory. "
                    "Did you forget to add missing hydrogens? "
                )
            traj_file = top_file if not traj_file else traj_file
            plot_id = self.plot_hbonds_over_time(hbond_counts, traj, traj_file)
            data_id = self.write_hbond_counts_to_file(hbond_counts, traj_file)
            return f"Hydrogen bond analysis completed. {data_id}, {plot_id} {rtrn_msg}."
        except Exception as e:
            return f"Error during hydrogen bond analysis: {e}"

    async def _arun(
        self,
        top_file: str,
        traj_file: str | None = None,
    ) -> str:
        raise NotImplementedError
