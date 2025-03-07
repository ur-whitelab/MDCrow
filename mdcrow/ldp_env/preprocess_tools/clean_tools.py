from typing import Dict, Optional

from openmm.app import PDBFile, PDBxFile
from pdbfixer import PDBFixer

# from ..path_registryutils.path_registry import PathRegistry
from state import MDCrowState
from utils import FileType


async def clean_pdb_file(
    pdb_id: str,
    state: MDCrowState,
    replace_nonstandard_residues: bool = True,
    add_missing_atoms: bool = True,
    remove_heterogens: bool = True,
    remove_water: bool = True,
    add_hydrogens: bool = True,
    add_hydrogens_ph: float = 7.0,
):
    """
    This tool performs various cleaning operations on a PDB or CIF file,
    including removing heterogens,
    adding missing atoms and hydrogens,
    replacing nonstandard residues, and/or removing water.

    Args:
        pdb_id (str): ID of the PDB/CIF file in the path registry.
        state (MDCrowState): The current state of the MDCrow environment.
        replace_nonstandard_residues (bool): Whether to replace nonstandard residues with standard ones.
        add_missing_atoms (bool): Whether to add missing atoms to the file from the SEQRES records.
        remove_heterogens (bool): Whether to remove heterogens from the file.
        remove_water (bool): Whether to remove water from the file.
        add_hydrogens (bool): Whether to add hydrogens to the file.
        add_hydrogens_ph (float): pH at which hydrogens are added.

    Returns:
        Dict[str, Optional[str]]: A dictionary containing the filename and file ID if successful, otherwise None.
    """
    path_registry = state.path_registry

    try:
        file_description = "Cleaned File: "
        try:
            pdbfile_path = path_registry.get_mapped_path(pdb_id)
            if "/" in pdbfile_path:
                pdbfile = pdbfile_path.split("/")[-1]
            else:
                pdbfile = pdbfile_path
            name, end = pdbfile.split(".")

        except Exception as e:
            print(f"error retrieving from path_registry, trying to read file {e}")
            return "Failed. File not found in path registry. ", 0, False
        print(f"file path: {pdbfile_path}")
        fixer = PDBFixer(filename=pdbfile_path)
        try:
            fixer.findMissingResidues()
        except Exception:
            print("error at findMissingResidues")
        try:
            fixer.findNonstandardResidues()
        except Exception:
            print("error at findNonstandardResidues")
        try:
            if remove_heterogens and remove_water:
                fixer.removeHeterogens(False)
                file_description += " Removed Heterogens, and Water Removed. "
            elif remove_heterogens and not remove_water:
                fixer.removeHeterogens(True)
                file_description += " Removed Heterogens, and Water Kept. "
        except Exception:
            print("Error at removeHeterogens")

        try:
            if replace_nonstandard_residues:
                fixer.replaceNonstandardResidues()
                file_description += " Replaced Nonstandard Residues. "
        except Exception:
            print("Error at replaceNonstandardResidues")
        try:
            fixer.findMissingAtoms()
        except Exception:
            print("Error at findMissingAtoms")
        try:
            if add_missing_atoms:
                fixer.addMissingAtoms()
        except Exception:
            print("Error at addMissingAtoms")
        try:
            if add_hydrogens:
                fixer.addMissingHydrogens(add_hydrogens_ph)
                file_description += f"Added Hydrogens at pH {add_hydrogens_ph}. "
        except Exception:
            print("Error at addMissingHydrogens")

        file_description += "Missing Atoms added and nonstandard residues replaced. "
        file_mode = "w" if add_hydrogens else "a"
        file_name = path_registry.write_file_name(
            type=FileType.PROTEIN,
            protein_name=name.split("_")[0],
            description="Clean",
            file_format=end,
        )
        file_id = path_registry.get_fileid(file_name, FileType.PROTEIN)
        directory = f"{path_registry.ckpt_pdb}"
        if end == "pdb":
            PDBFile.writeFile(
                fixer.topology,
                fixer.positions,
                open(f"{directory}/{file_name}", file_mode),
            )
        elif end == "cif":
            PDBxFile.writeFile(
                fixer.topology,
                fixer.positions,
                open(f"{directory}/{file_name}", file_mode),
            )

        path_registry.map_path(file_id, f"{directory}/{file_name}", file_description)
        return f"Succeeded. File cleaned! \nFile ID: {file_id}", 0, False
    except FileNotFoundError as e:
        return "Failed. Check your file path. File not found: " + str(e), 0, False
    except Exception as e:
        print(e)
        return f"Failed. {type(e).__name__}: {e}", 0, False
