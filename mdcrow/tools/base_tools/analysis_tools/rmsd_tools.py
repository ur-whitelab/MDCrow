from typing import Optional, Type

import matplotlib.pyplot as plt
import mdtraj as md
from langchain.tools import BaseTool
from pydantic import BaseModel, Field

from mdcrow.utils import PathRegistry, load_traj_with_ref, save_plot, save_to_csv


def rmsd(path_registry, traj, ref_traj, mol_name, select="protein"):
    """
    Calculate the root mean square deviation (RMSD) of each selected atom.
    Can be used for either protions or small molecules.
    """
    print("Calculating RMSD...")
    msg = ""
    idx = traj.topology.select(select)
    if len(idx) == 0:
        if select == "protein":
            # sometimes it's small molecules, so no residues
            residues = set([residue.name for residue in traj.topology.residues])
            print(f"No atoms found for selection '{select}'. ")
            print(f"Only residues found in the file(s) are {residues}.")
            print("Currently trying 'all' atoms.")
            select = "all"
            idx = traj.topology.select("all")
        else:
            raise ValueError(f"No atoms found for selection '{select}'.")

    rmsd_select = md.rmsd(traj, ref_traj, atom_indices=idx)

    if rmsd_select.shape[0] == 1:
        # RMSD is a single value
        return f"RMSD calculated. {rmsd_select[0]:.5f} nm"

    analysis = f"rmsd_{mol_name}"
    csv_file_id = save_to_csv(
        path_registry,
        rmsd_select,
        analysis,
        description=f"RMSD for {mol_name}",
        header="RMSD (nm)",
    )

    # plot rmsd
    fig, ax = plt.subplots()
    ax.plot(rmsd_select, label=select)
    ax.legend()
    ax.set(
        **{
            "xlabel": "time",
            "ylabel": "RMSD / nm",
        }
    )
    fig_id = save_plot(path_registry, analysis, f"RMSD plot for {mol_name}")
    plt.close()
    msg = (
        f"RMSD calculated and saved to csv with file ID {csv_file_id}. "
        f"Plot saved with plot ID {fig_id}. "
    )
    return msg


def rmsf(path_registry, traj, ref_traj, mol_name, select="protein"):
    """
    Calculate the root mean square fluctuation (RMSF) of each selected atom.
    Usually used for proteins, but can select 'all' for small molecules.
    """
    print("Calculating RMSF...")
    idx = traj.topology.select(select)
    if len(idx) == 0:
        residues = set([residue.name for residue in traj.topology.residues])
        raise ValueError(
            (
                f"No atoms found for selection '{select}'. "
                f"Only residues found in the file(s) are {residues}. \n"
                "Try 'all' for small molecules. \n"
            )
        )

    rmsf = md.rmsf(traj, ref_traj)
    rmsf_select = rmsf[idx]
    analysis = f"rmsf_{mol_name}"
    csv_file_id = save_to_csv(
        path_registry,
        rmsf_select,
        analysis,
        description=f"RMSF for {mol_name}",
        header="RMSF (nm)",
    )

    # plot rmsf
    fig, ax = plt.subplots()

    # check if whether to plot protein or all atoms
    protein_mode = True
    select_idx = traj.topology.select(f"{select} and name CA")
    if len(select_idx) == 0:
        protein_mode = False
        select_idx = idx

    xlabel = "Residue Number" if protein_mode else "Atom Number"
    title = (
        "RMS Fluctuation (carbon alpha)"
        if protein_mode
        else "RMS Fluctuation (all atoms)"
    )

    ax.plot(select_idx, rmsf[select_idx], label=select)
    ax.legend()
    ax.set(
        **{
            "xlabel": xlabel,
            "ylabel": "RMSF / nm",
            "title": title,
        }
    )
    fig_id = save_plot(path_registry, analysis, f"RMSF plot for {mol_name}")
    plt.close()
    msg = (
        f"RMSF calculated and saved to csv with file ID {csv_file_id}. "
        f"Plot saved with plot ID {fig_id}. "
    )
    return msg


def lprmsd(path_registry, traj, ref_traj, mol_name, select="protein"):
    """
    LP-RMSD is Linear-Programming Root-Mean-Squared Deviation.
    It gives the global minimum of the means squared deviation using
    3-step optimization process. More info in MDTraj docs.

    Note:
    - LPRMSD is really useful for when you have indistinguishable atoms that
    you want to permute and calculate the minimum distance under permutations.
    - Example: atoms with exchange symmetry like multiple water molecules.
    """
    print(f"Calculating LP-RMSD for with select '{select}'...")
    idx = traj.topology.select(select)
    if len(idx) == 0:
        residues = set([residue.name for residue in traj.topology.residues])
        raise ValueError(
            (
                f"No atoms found for selection '{select}'. "
                f"Only residues found in the file(s) are {residues}. \n"
                "Try 'all' for small molecules. \n"
            )
        )

    lprmsd = md.rmsd(traj, ref_traj, atom_indices=idx)

    if lprmsd.shape[0] == 1:
        # RMSD is a single value
        return f"Linear Programming RMSD calculated. {lprmsd[0]:.5f} nm"

    csv_file_id = save_to_csv(
        path_registry,
        lprmsd,
        f"lprmsd_{mol_name}",
        description=f"LP-RMSD for {mol_name}",
        header="LP-RMSD (nm)",
    )
    return f"LP-RMSD calculated and saved to csv with file ID {csv_file_id}"


class RMSDInputSchema(BaseModel):
    top_id: str = Field(None, description="File ID for the topology file.")
    traj_id: Optional[str] = Field(
        None,
        description=(
            "File ID for the trajectory file. "
            "Required for RMSF. Otherwise, optional."
        ),
    )
    ref_top_id: Optional[str] = Field(
        None,
        description=(
            "File ID for the topology file as reference. "
            "Only provide if it's different from target"
        ),
    )
    ref_traj_id: Optional[str] = Field(
        None,
        description=(
            "File ID for the trajectory file as reference. "
            "Only provide if it's different from target"
        ),
    )
    select: Optional[str] = Field(
        "protein",
        description=(
            "atom selection following MDTraj syntax for topology.select. "
            "Examples are 'protein', 'backbone', 'sidechain'"
        ),
    )
    mol_name: Optional[str] = Field(
        None, description="Name of the molecule or protein."
    )


class ComputeRMSD(BaseTool):
    name: str = "ComputeRMSD"
    description: str = (
        "Compute root mean square deviation (RMSD) of all "
        "conformations in target to a reference conformation."
    )
    args_schema: Type[BaseModel] = RMSDInputSchema
    path_registry: PathRegistry | None

    def __init__(self, path_registry=None):
        super().__init__()
        self.path_registry = path_registry

    def _run(
        self,
        top_id,
        traj_id=None,
        ref_top_id=None,
        ref_traj_id=None,
        mol_name=None,
        select="protein",
    ):
        try:
            if mol_name is None:
                mol_name = top_id.replace("top_sim0_", "")
            traj, ref_traj = load_traj_with_ref(
                self.path_registry, top_id, traj_id, ref_top_id, ref_traj_id
            )
            msg = rmsd(self.path_registry, traj, ref_traj, mol_name, select)
            return f"Succeeded. {msg}"
        except Exception as e:
            return f"Failed. {type(e).__name__}: {e}"


class ComputeRMSF(BaseTool):
    name: str = "ComputeRMSF"
    description: str = (
        "Compute root mean square fluctuation (RMSF) of all "
        "conformations in target to a reference conformation"
    )
    args_schema: Type[BaseModel] = RMSDInputSchema
    path_registry: PathRegistry | None

    def __init__(self, path_registry=None):
        super().__init__()
        self.path_registry = path_registry

    def _run(
        self,
        top_id,
        traj_id=None,
        ref_top_id=None,
        ref_traj_id=None,
        mol_name=None,
        select="protein",
    ):
        try:
            if mol_name is None:
                mol_name = top_id.replace("top_sim0_", "")
            if ref_top_id is None or ref_top_id == top_id:  # if there's no ref
                traj_required = True
            else:
                traj_required = False

            traj, ref_traj = load_traj_with_ref(
                self.path_registry,
                top_id,
                traj_id,
                ref_top_id,
                ref_traj_id,
                traj_required=traj_required,
            )
            msg = rmsf(self.path_registry, traj, ref_traj, mol_name, select)
            return f"Succeeded. {msg}"
        except Exception as e:
            return f"Failed. {type(e).__name__}: {e}"


class ComputeLPRMSD(BaseTool):
    name: str = "ComputeLP-RMSD"
    description: str = (
        # description from mdtraj docs
        "Compute Linear-Programming Root-Mean-Squared Deviation "
        "(LP-RMSD) of all conformations in target to a reference "
        "conformation. The LP-RMSD is the minimum RMSD between "
        "two sets of points, minimizing over both the rotational/"
        "translational degrees of freedom AND the label "
        "correspondences between points in the target and "
        "reference conformations."
    )
    args_schema: Type[BaseModel] = RMSDInputSchema
    path_registry: PathRegistry | None

    def __init__(self, path_registry=None):
        super().__init__()
        self.path_registry = path_registry

    def _run(
        self,
        top_id,
        traj_id=None,
        ref_top_id=None,
        ref_traj_id=None,
        mol_name=None,
        select="protein",
    ):
        try:
            if mol_name is None:
                mol_name = top_id.replace("top_sim0_", "")
            traj, ref_traj = load_traj_with_ref(
                self.path_registry, top_id, traj_id, ref_top_id, ref_traj_id
            )
            msg = lprmsd(self.path_registry, traj, ref_traj, mol_name, select)
            return f"Succeeded. {msg}"
        except Exception as e:
            return f"Failed. {type(e).__name__}: {e}"
