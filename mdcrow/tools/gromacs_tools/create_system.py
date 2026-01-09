import os
import subprocess
from typing import Optional

from langchain.tools import BaseTool
from pydantic import BaseModel, Field

from mdcrow.utils import FileType, PathRegistry


class PrepareGromacsProteinInput(BaseModel):
    protein_file_id: str = Field(
        ..., description="File ID of the protein PDB in the path registry"
    )
    forcefield: str = Field(
        default="amber99sb-ildn", description="GROMACS force field name"
    )
    water_model: str = Field(default="tip3p", description="Water model")
    box_padding_nm: float = Field(
        default=1.0, description="Padding distance to box edge (nm)"
    )


class PrepareGromacsProteinSystemTool(BaseTool):
    name = "PrepareGromacsProteinSystem"
    description = (
        "Prepare a protein-only system for GROMACS by assigning force fields, "
        "adding hydrogens, solvating, and neutralizing the system."
    )
    args_schema = PrepareGromacsProteinInput

    path_registry: Optional[PathRegistry]

    def __init__(self, path_registry: PathRegistry):
        super().__init__()
        self.path_registry = path_registry

    def _run(
        self,
        protein_file_id: str,
        forcefield: str = "amber99sb-ildn",
        water_model: str = "tip3p",
        box_padding_nm: float = 1.0,
    ) -> str:

        pdb_path = self.path_registry.get_mapped_path(protein_file_id)
        if not os.path.exists(pdb_path):
            return f"Failed. PDB file not found for ID {protein_file_id}"

        workdir = self.path_registry.ckpt_dir + "/gromacs_preparation"
        os.makedirs(workdir, exist_ok=True)
        os.chdir(workdir)

        subprocess.run(
            [
                "gmx",
                "pdb2gmx",
                "-f",
                pdb_path,
                "-o",
                "processed.gro",
                "-p",
                "topol.top",
                "-ff",
                forcefield,
                "-water",
                water_model,
                "-ignh",
            ],
            check=True,
        )

        subprocess.run(
            [
                "gmx",
                "editconf",
                "-f",
                "processed.gro",
                "-o",
                "boxed.gro",
                "-c",
                "-d",
                str(box_padding_nm),
                "-bt",
                "cubic",
            ],
            check=True,
        )

        subprocess.run(
            [
                "gmx",
                "solvate",
                "-cp",
                "boxed.gro",
                "-cs",
                "spc216.gro",
                "-o",
                "solvated.gro",
                "-p",
                "topol.top",
            ],
            check=True,
        )

        with open("ions.mdp", "w") as f:
            f.write("integrator = steep\nnsteps = 500\n")

        subprocess.run(
            [
                "gmx",
                "grompp",
                "-f",
                "ions.mdp",
                "-c",
                "solvated.gro",
                "-p",
                "topol.top",
                "-o",
                "ions.tpr",
                "-maxwarn",
                "1",
            ],
            check=True,
        )

        subprocess.run(
            [
                "gmx",
                "genion",
                "-s",
                "ions.tpr",
                "-o",
                "solv_ions.gro",
                "-p",
                "topol.top",
                "-neutral",
            ],
            input=b"SOL\n",
            check=True,
        )
        coord_name = self.path_registry.write_file_name(
            type=FileType.RECORD,
            record_type="GROMACS_COORD",
            protein_file_id=protein_file_id,
            file_format="gro",
        )

        top_name = self.path_registry.write_file_name(
            type=FileType.RECORD,
            record_type="GROMACS_TOP",
            protein_file_id=protein_file_id,
            file_format="top",
        )

        os.rename("solv_ions.gro", coord_name)
        os.rename("topol.top", top_name)

        coord_id = self.path_registry.get_fileid(coord_name, FileType.RECORD)
        self.path_registry.map_path(
            coord_id, f"{workdir}/{coord_name}", "GROMACS solvated coordinates"
        )

        top_id = self.path_registry.get_fileid(top_name, FileType.RECORD)
        self.path_registry.map_path(
            top_id, f"{workdir}/{top_name}", "GROMACS topology file"
        )

        return (
            "Succeeded. GROMACS system prepared.\n"
            f"Coordinates ID: {coord_id}\nTopology ID: {top_id}"
        )
