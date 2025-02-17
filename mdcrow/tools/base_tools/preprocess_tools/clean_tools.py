from typing import Optional, Type

from langchain.tools import BaseTool
from openmm.app import PDBFile, PDBxFile
from pdbfixer import PDBFixer
from pydantic import BaseModel, Field

from mdcrow.utils import FileType, PathRegistry


class CleaningToolFunctionInput(BaseModel):
    """Input model for CleaningToolFunction"""

    pdb_id: str = Field(..., description="ID of the pdb/cif file in the path registry")
    replace_nonstandard_residues: bool = Field(
        True, description="Whether to replace nonstandard residues with standard ones. "
    )
    add_missing_atoms: bool = Field(
        True,
        description="Whether to add missing atoms to the file from the SEQRES records.",
    )
    remove_heterogens: bool = Field(
        True, description="Whether to remove heterogens from the file."
    )
    remove_water: bool = Field(
        True,
        description="""Whether to remove water from the file.
        remove_heterogens must be True.""",
    )
    add_hydrogens: bool = Field(
        True, description="Whether to add hydrogens to the file."
    )
    add_hydrogens_ph: float = Field(7.0, description="pH at which hydrogens are added.")


class CleaningToolFunction(BaseTool):
    name = "CleaningToolFunction"
    description = """
    This tool performs various cleaning operations on a PDB or CIF file.
    Operations can include removing heterogens,
    adding missing atoms and hydrogens,
    replacing nonstandard residues, and/or removing water.

    """
    args_schema: Type[BaseModel] = CleaningToolFunctionInput

    path_registry: Optional[PathRegistry]

    def __init__(self, path_registry):
        super().__init__()
        self.path_registry = path_registry

    def _run(self, **input_args) -> str:
        """Use the tool with specified operations."""
        if self.path_registry is None:
            return "Path registry not initialized"
        try:
            if "input_args" in input_args.keys():
                input_args = input_args["input_args"]
            else:
                input_args = input_args
            pdbfile_id = input_args.get("pdb_id", None)
            if pdbfile_id is None:
                return """Failed. No file was provided.
                The input has to be a dictionary with the key 'pdb_id'"""
            remove_heterogens = input_args.get("remove_heterogens", True)
            remove_water = input_args.get("remove_water", True)
            add_hydrogens = input_args.get("add_hydrogens", True)
            add_hydrogens_ph = input_args.get("add_hydrogens_ph", 7.0)
            replace_nonstandard_residues = input_args.get(
                "replace_nonstandard_residues", True
            )
            add_missing_atoms = input_args.get("add_missing_atoms", True)
            input_args.get("output_path", None)

            if self.path_registry is None:
                return "Failed. Path registry not initialized"
            file_description = "Cleaned File: "
            try:
                pdbfile_path = self.path_registry.get_mapped_path(pdbfile_id)
                if "/" in pdbfile_path:
                    pdbfile = pdbfile_path.split("/")[-1]
                else:
                    pdbfile = pdbfile_path
                name, end = pdbfile.split(".")

            except Exception as e:
                print(f"error retrieving from path_registry, trying to read file {e}")
                return "Failed. File not found in path registry. "
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
                print("error at removeHeterogens")

            try:
                if replace_nonstandard_residues:
                    fixer.replaceNonstandardResidues()
                    file_description += " Replaced Nonstandard Residues. "
            except Exception:
                print("error at replaceNonstandardResidues")
            try:
                fixer.findMissingAtoms()
            except Exception:
                print("error at findMissingAtoms")
            try:
                if add_missing_atoms:
                    fixer.addMissingAtoms()
            except Exception:
                print("error at addMissingAtoms")
            try:
                if add_hydrogens:
                    fixer.addMissingHydrogens(add_hydrogens_ph)
                    file_description += f"Added Hydrogens at pH {add_hydrogens_ph}. "
            except Exception:
                print("error at addMissingHydrogens")

            file_description += (
                "Missing Atoms Added and replaces nonstandard residues. "
            )
            file_mode = "w" if add_hydrogens else "a"
            file_name = self.path_registry.write_file_name(
                type=FileType.PROTEIN,
                protein_name=name.split("_")[0],
                description="Clean",
                file_format=end,
            )
            file_id = self.path_registry.get_fileid(file_name, FileType.PROTEIN)
            directory = f"{self.path_registry.ckpt_pdb}"
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

            self.path_registry.map_path(
                file_id, f"{directory}/{file_name}", file_description
            )
            return f"Succeeded. File cleaned! \nFile ID: {file_id}"
        except FileNotFoundError as e:
            return "Failed. Check your file path. File not found: " + str(e)
        except Exception as e:
            print(e)
            return f"Failed. {type(e).__name__}: {e}"

    async def _arun(
        self, query: str, remove_water: bool = False, add_hydrogens: bool = False
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("Asynchronous operation not supported yet.")
