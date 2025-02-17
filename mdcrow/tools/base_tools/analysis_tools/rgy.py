from typing import Optional

import matplotlib.pyplot as plt
import mdtraj as md
import numpy as np
from langchain.tools import BaseTool

from mdcrow.utils import FileType, PathRegistry, load_single_traj


class RadiusofGyration:
    def __init__(self, path_registry):
        self.path_registry = path_registry
        self.top_file = ""
        self.traj_file = ""
        self.traj = None
        self.rgy_file = ""

    def _load_traj(self, top_file: str, traj_file: str):
        self.traj_file = traj_file
        self.top_file = top_file
        self.traj = load_single_traj(
            path_registry=self.path_registry,
            top_fileid=top_file,
            traj_fileid=traj_file,
            traj_required=True,
        )

    def rgy_per_frame(self) -> str:
        rg_per_frame = md.compute_rg(self.traj)
        self.rgy_file = (
            f"{self.path_registry.ckpt_figures}/radii_of_gyration_{self.traj_file}.csv"
        )
        rgy_id = f"rgy_{self.traj_file}"
        np.savetxt(
            self.rgy_file,
            rg_per_frame,
            delimiter=",",
            header="Radius of Gyration (nm)",
        )
        self.path_registry.map_path(
            f"rgy_{self.traj_file}",
            self.rgy_file,
            description=f"Radii of gyration per frame for {self.traj_file}",
        )
        return f"Radii of gyration saved to {self.rgy_file} with id {rgy_id}."

    def rgy_average(self) -> str:
        if not self.rgy_file:
            _ = self.rgy_per_frame()
        rg_per_frame = np.loadtxt(self.rgy_file, delimiter=",", skiprows=1)
        avg_rg = rg_per_frame.mean()

        return f"Average radius of gyration: {avg_rg:.2f} nm"

    def plot_rgy(self) -> str:
        if not self.rgy_file:
            _ = self.rgy_per_frame()
        rg_per_frame = np.loadtxt(self.rgy_file, delimiter=",", skiprows=1)
        fig_analysis = f"rgy_{self.traj_file}"
        plot_name = self.path_registry.write_file_name(
            type=FileType.FIGURE, fig_analysis=fig_analysis, file_format="png"
        )
        print("plot_name: ", plot_name)
        plot_id = self.path_registry.get_fileid(
            file_name=plot_name, type=FileType.FIGURE
        )
        plot_path = f"{self.path_registry.ckpt_figures}/{plot_name}"
        plot_path = plot_path if plot_path.endswith(".png") else plot_path + ".png"
        print("plot_path", plot_path)
        plt.plot(rg_per_frame)
        plt.xlabel("Frame")
        plt.ylabel("Radius of Gyration (nm)")
        plt.title(f"{self.traj_file} - Radius of Gyration Over Time")

        plt.savefig(f"{plot_path}")
        self.path_registry.map_path(
            plot_id,
            plot_path,
            description=f"Plot of radii of gyration over time for {self.traj_file}",
        )
        plt.close()
        plt.clf()
        return "Plot saved as: " + f"{plot_name} with plot ID {plot_id}"

    def compute_plot_return_avg(self) -> str:
        rgy_per_frame = self.rgy_per_frame()
        avg_rgy = self.rgy_average()
        plot_rgy = self.plot_rgy()
        return rgy_per_frame + plot_rgy + avg_rgy


class RadiusofGyrationTool(BaseTool):
    name = "RadiusofGyrationTool"
    description = """This tool calculates and plots
    the radius of gyration
    at each frame of a given trajectory and retuns the average.
    Give this tool BOTH the trajectory file ID and the
    topology file ID."""

    path_registry: Optional[PathRegistry]
    rgy: Optional[RadiusofGyration]
    load_traj: bool = True

    def __init__(self, path_registry, load_traj=True):
        super().__init__()
        self.path_registry = path_registry
        self.rgy = RadiusofGyration(path_registry)
        self.load_traj = load_traj  # only for testing

    def _run(self, traj_file: str, top_file: str) -> str:
        """use the tool."""
        assert self.rgy is not None, "RadiusofGyration instance is not initialized"

        if self.load_traj:
            try:
                self.rgy._load_traj(top_file=top_file, traj_file=traj_file)
            except Exception as e:
                return f"Error loading traj: {e}"
        try:
            return "Succeeded. " + self.rgy.compute_plot_return_avg()
        except Exception as e:
            return f"Failed Computing RGY: {e}"

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")
