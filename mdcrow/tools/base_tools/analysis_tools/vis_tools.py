import os
import shutil
import subprocess
from typing import Optional

import nbformat as nbf
from langchain.tools import BaseTool

from mdcrow.utils import PathRegistry


class VisFunctions:
    def __init__(self, path_registry):
        self.path_registry = path_registry
        self.starting_files = os.listdir(".")

    def _find_png(self):
        current_files = os.listdir(".")
        new_files = [f for f in current_files if f not in self.starting_files]
        png_files = [f for f in new_files if f.endswith(".png")]
        return png_files

    def run_molrender(self, cif_path: str) -> str:
        """Function to run molrender,
        it requires node.js to be installed
        and the molrender package to be
        installed globally.
        This will save .png
        files in the current
        directory."""
        self.cif_file_name = os.path.basename(cif_path)

        cmd = ["molrender", "all", cif_path, ".", "--format", "png"]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
        except subprocess.CalledProcessError:
            raise RuntimeError("molrender package not found")
        file_name = self._find_png()
        if not file_name:
            raise FileNotFoundError("No .png files were created")
        shutil.move(file_name[0], f"{self.path_registry.ckpt_figures}/{file_name[0]}")
        self.path_registry.map_path(
            f"mol_render_{self.cif_file_name}",
            f"{self.path_registry.ckpt_figures}/{file_name[0]}",
            (
                f"Visualization of cif file {self.cif_file_name}"
                "as png file. using molrender."
            ),
        )

        if result.returncode != 0:
            raise RuntimeError(f"Error running molrender: {result.stderr}")
        else:
            print(f"Output: {result.stdout}")
        return (
            "Visualization using molrender complete, "
            f"saved as: mol_render_{self.cif_file_name}"
        )

    def create_notebook(self, cif_file: str) -> str:
        """This is for plan B
        tool, it will create
        a notebook
        with the code to
        install nglview and
        display the cif/pdb file."""
        self.cif_file_name = os.path.basename(cif_file)

        # Create a new notebook
        nb = nbf.v4.new_notebook()

        # Code to install NGLview
        install_code = "!pip install -q nglview"

        # Code to import NGLview and display a file
        import_code = f"""
import nglview as nv
view = nv.show_file("{cif_file}")
view
"""

        # Create new code cells
        install_cell = nbf.v4.new_code_cell(source=install_code)
        import_cell = nbf.v4.new_code_cell(source=import_code)

        # Add the cells
        nb.cells.extend([install_cell, import_cell])

        # Write the notebook to a file
        notebook_name = (
            f"{self.path_registry.ckpt_figures}"
            f"/{self.cif_file_name.split('.')[0]}_vis.ipynb"
        )
        with open(notebook_name, "w") as f:
            nbf.write(nb, f)

        self.path_registry.map_path(
            notebook_name,
            notebook_name,
            f"Notebook to visualize cif/pdb file {self.cif_file_name} using nglview.",
        )
        return "Visualization Complete"


class VisualizeProtein(BaseTool):
    """To get a png, you must install molrender
    https://github.com/molstar/molrender/tree/master
    Otherwise, you will get a notebook where you
    can visualize the protein."""

    name = "PDBVisualization"
    description = """This tool will create
                    a visualization of a cif
                    file as a png file OR
                    it will create
                    a .ipynb file with the
                    visualization of the
                    file, depending on the
                    packages available.
                    If a notebook is created,
                    the user can open the
                    notebook and visualize the
                    system."""
    path_registry: Optional[PathRegistry]

    def __init__(self, path_registry: Optional[PathRegistry]):
        super().__init__()
        self.path_registry = path_registry

    def _run(self, cif_file_name: str) -> str:
        """use the tool."""
        if not self.path_registry:
            return "Failed. Error: Path registry is not set"
        cif_path = self.path_registry.get_mapped_path(cif_file_name)
        if not cif_path:
            return f"Failed. File not found: {cif_file_name}"
        vis = VisFunctions(self.path_registry)
        try:
            return "Succeeded. " + vis.run_molrender(cif_path)
        except (RuntimeError, FileNotFoundError) as e:
            print(f"Error running molrender: {str(e)}. Using NGLView instead.")
            try:
                vis.create_notebook(cif_path)
                return "Succeeded. Visualization created as notebook"
            except Exception as e:
                return f"Failed. {type(e).__name__}: {e}"

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")
