import os
import subprocess
from typing import Optional

from langchain.tools import BaseTool
from pydantic import BaseModel, Field

from mdcrow.utils import FileType, PathRegistry


class RunGromacsMDInput(BaseModel):
    topology_id: str
    coordinates_id: str
    ensemble: str = Field(description="NVT or NPT")
    temperature: float = Field(default=300.0)
    nsteps: int = Field(default=500_000)


class RunGromacsMDTool(BaseTool):
    name = "RunGromacsMD"
    description = (
        "Run a GROMACS MD simulation using NVT or NPT."
        "Requires topology (.top) and coordinate (.gro) files."
    )
    args_schema = RunGromacsMDInput

    path_registry: Optional[PathRegistry]

    def __init__(self, path_registry: PathRegistry):
        super().__init__()
        self.path_registry = path_registry

    def _run(self, topology_id, coordinates_id, ensemble, temperature, nsteps):

        top_path = self.path_registry.get_mapped_path(topology_id)
        gro_path = self.path_registry.get_mapped_path(coordinates_id)

        if not os.path.exists(top_path):
            return f"Failed. Topology file not found: {top_path}"
        if not os.path.exists(gro_path):
            return f"Failed. Coordinate file not found: {gro_path}"

        workdir = self.path_registry.ckpt_simulations
        os.chdir(workdir)

        for stale in ["md.xtc", "md.gro"]:
            if os.path.exists(stale):
                os.remove(stale)

        with open("minim.mdp", "w") as f:
            f.write("integrator = steep\nnsteps = 5000\n")

        try:
            subprocess.run(
                [
                    "gmx",
                    "grompp",
                    "-f",
                    "minim.mdp",
                    "-c",
                    gro_path,
                    "-p",
                    top_path,
                    "-o",
                    "em.tpr",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            subprocess.run(
                ["gmx", "mdrun", "-deffnm", "em"],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            return f"Failed during energy minimization: {e.stderr}"

        # just explicitly setting many of these. we could make this more flexible later
        if ensemble.upper() == "NVT":
            mdp = f"""
integrator = md
nsteps = {nsteps}
dt = 0.002
tcoupl = V-rescale
tc-grps = Protein Non-Protein
tau_t = 0.1 0.1
ref_t = {temperature} {temperature}
constraints = h-bonds
cutoff-scheme = Verlet
nstxout-compressed = 500
compressed-x-grps = System
"""
        elif ensemble.upper() == "NPT":
            mdp = f"""
integrator = md
nsteps = {nsteps}
dt = 0.002
tcoupl = V-rescale
tc-grps = Protein Non-Protein
tau_t = 0.1 0.1
ref_t = {temperature} {temperature}
pcoupl = Parrinello-Rahman
pcoupltype = isotropic
tau_p = 2.0
ref_p = 1.0
compressibility = 4.5e-5
constraints = h-bonds
cutoff-scheme = Verlet
nstxout-compressed = 500
compressed-x-grps = System
"""
        else:
            return f"Failed. Unsupported ensemble: {ensemble}"

        with open("md.mdp", "w") as f:
            f.write(mdp)

        try:
            subprocess.run(
                [
                    "gmx",
                    "grompp",
                    "-f",
                    "md.mdp",
                    "-c",
                    "em.gro",
                    "-p",
                    top_path,
                    "-o",
                    "md.tpr",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            subprocess.run(
                ["gmx", "mdrun", "-deffnm", "md", "-x", "md.xtc"],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            return f"Failed during MD simulation: {e.stderr}"

        traj_name = self.path_registry.write_file_name(
            type=FileType.RECORD,
            record_type="TRAJ",
            file_format="xtc",
        )
        gro_name = self.path_registry.write_file_name(
            type=FileType.RECORD,
            record_type="FINAL",
            file_format="gro",
        )

        if not os.path.exists("md.xtc"):
            return "Failed. GROMACS did not produce md.xtc. Check md.log for errors."
        if not os.path.exists("md.gro"):
            return "Failed. GROMACS did not produce md.gro. Check md.log for errors."

        os.rename("md.xtc", traj_name)
        os.rename("md.gro", gro_name)

        traj_id = self.path_registry.get_fileid(traj_name, FileType.RECORD)
        self.path_registry.map_path(traj_id, f"{workdir}/{traj_name}", "MD trajectory")

        gro_id = self.path_registry.get_fileid(gro_name, FileType.RECORD)
        self.path_registry.map_path(gro_id, f"{workdir}/{gro_name}", "Final structure")

        return (
            f"Succeeded. MD completed using {ensemble} ensemble.\n"
            f"Trajectory ID: {traj_id}\nFinal structure ID: {gro_id}"
        )
