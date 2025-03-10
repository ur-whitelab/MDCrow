import csv
import re

import matplotlib.pyplot as plt
from state import MDCrowState
from utils import FileType


class PlottingTools:
    def __init__(
        self,
        path_registry,
    ):
        self.path_registry = path_registry
        self.data = None
        self.headers = None
        self.matched_headers = None
        self.file_id = None
        self.file_path = None

    def _find_file(self, file_id: str) -> None:
        self.file_id = file_id
        self.file_path = self.path_registry.get_mapped_path(file_id)
        if not self.file_path:
            raise FileNotFoundError("File not found.")
        return None

    def process_csv(self) -> None:
        with open(self.file_path, "r") as f:
            reader = csv.DictReader(f)
            self.headers = reader.fieldnames if reader.fieldnames is not None else []
            self.data = list(reader)

        self.matched_headers = [
            (i, header)
            for i, header in enumerate(self.headers)
            if re.search(r"(step|time)", header, re.IGNORECASE)
        ]

        if not self.matched_headers or not self.headers or not self.data:
            raise ValueError("File could not be processed.")
        return None

    def plot_data(self) -> str:
        if self.matched_headers:
            time_or_step = self.matched_headers[0][1]
            xlab = "step" if "step" in time_or_step.lower() else "time"
        else:
            raise ValueError("No timestep found.")

        failed_headers = []
        created_plots = []
        for header in self.headers:
            if header != time_or_step:
                try:
                    x = [float(row[time_or_step]) for row in self.data]
                    y = [float(row[header]) for row in self.data]

                    header_lab = (
                        header.split("(")[0].strip() if "(" in header else header
                    ).lower()
                    # Generate and save the plot
                    plt.figure()
                    plt.plot(x, y)
                    plt.xlabel(xlab)
                    plt.ylabel(header)
                    plt.title(f"{self.file_id}_{xlab} vs {header_lab}")
                    fig_vs = f"{xlab}vs{header_lab}"
                    plot_name = self.path_registry.write_file_name(
                        type=FileType.FIGURE,
                        Log_id=self.file_id,
                        fig_analysis=fig_vs,
                        file_format="png",
                    )
                    plot_id = self.path_registry.get_fileid(
                        file_name=plot_name, type=FileType.FIGURE
                    )
                    plt.savefig(f"{self.path_registry.ckpt_figures}/{plot_name}")
                    self.path_registry.map_path(
                        plot_id,
                        f"{self.path_registry.ckpt_figures}/{plot_name}",
                        (
                            f"Post Simulation Figure for {self.file_id}"
                            f" - {header_lab} vs {xlab}"
                        ),
                    )
                    plt.close()

                    created_plots.append(plot_name + "with ID: " + plot_id)
                except ValueError:
                    failed_headers.append(header)

        if (
            len(failed_headers) == len(self.headers) - 1
        ):  # -1 to account for time_or_step header
            raise Exception("All plots failed due to non-numeric data.")

        return ", ".join(created_plots)


def get_simulation_figure(
    state: MDCrowState,
    file_id: str,
):
    """
    This tool will take a csv file id output from an openmm simulation and create \
    figures for all physical parameters versus timestep of the simulation. \
    Give this tool the name of the csv file output from the simulation. \
    The tool will get the exact path.

    Args:
        file_id: File ID of the csv file from which to make the plot
    """

    try:
        plotting_tools = PlottingTools(state.path_registry)
        plotting_tools._find_file(file_id)
        plotting_tools.process_csv()
        plot_result = plotting_tools.plot_data()
        if isinstance(plot_result, str):
            return "Succeeded. IDs of figures created: " + plot_result, 0, False
        else:
            return "Failed. No figures created.", 0, False
    except Exception as e:
        return "Failed. " + str(e), 0, False
