import os
import re
import subprocess
import typing
from typing import Any, Dict, List, Type, Union

from langchain.tools import BaseTool
from pydantic import BaseModel, Field, ValidationError

from mdagent.utils import PathRegistry, validate_tool_args

from .pdb_fix import Validate_Fix_PDB
from .pdb_get import MolPDB


def summarize_errors(errors):
    error_summary = {}

    # Regular expression pattern to capture the error type and line number
    pattern = r"\[!\] Offending field \((.+?)\) at line (\d+)"

    for error in errors:
        match = re.search(pattern, error)
        if match:
            error_type, line_number = match.groups()
            # If this error type hasn't been seen before,
            # initialize it in the dictionary
            if error_type not in error_summary:
                error_summary[error_type] = {"lines": []}
            error_summary[error_type]["lines"].append(line_number)

    # Format the summarized errors for display
    summarized_strings = []
    for error_type, data in error_summary.items():
        line_count = len(data["lines"])
        if line_count > 3:
            summarized_strings.append(f"{error_type}: total {line_count} lines")
        else:
            summarized_strings.append(f"{error_type}: lines: {','.join(data['lines'])}")

    return summarized_strings


class Molecule:
    def __init__(self, filename, file_id, number_of_molecules=1, instructions=None):
        self.filename = filename
        self.id = file_id
        self.number_of_molecules = number_of_molecules
        self.instructions = instructions if instructions else []
        self.load()

    def load(self):
        # load the molecule data (optional)
        pass

    def get_number_of_atoms(self):
        # return the number of atoms in this molecule
        pass


class PackmolBox:
    def __init__(
        self,
        path_registry,
        file_number=1,
        file_description="PDB file for simulation with: \n",
    ):
        self.path_registry = path_registry
        self.molecules = []
        self.file_number = file_number
        self.file_description = file_description
        self.final_name = None

    def add_molecule(self, molecule: Molecule) -> None:
        self.molecules.append(molecule)
        self.file_description += f"""{molecule.number_of_molecules} of
        {molecule.filename} as {molecule.instructions} \n"""
        return None

    def generate_input_header(self) -> None:
        # Generate the header of the input file in .inp format
        orig_pdbs_ids = [
            f"{molecule.number_of_molecules}_{molecule.id}"
            for molecule in self.molecules
        ]

        _final_name = f'{"_and_".join(orig_pdbs_ids)}'

        self.file_description = (
            "Packed Structures of the following molecules:\n"
            + "\n".join(
                [
                    f"Molecule ID: {molecule.id}, "
                    f"Number of Molecules: {molecule.number_of_molecules}"
                    for molecule in self.molecules
                ]
            )
        )
        while os.path.exists(
            f"{self.path_registry.ckpt_files}/pdb/{_final_name}_v{self.file_number}.pdb"
        ):
            self.file_number += 1

        self.final_name = f"{_final_name}_v{self.file_number}.pdb"
        with open("packmol.inp", "w") as out:
            out.write("##Automatically generated by LangChain\n")
            out.write("tolerance 2.0\n")
            out.write("filetype pdb\n")
            out.write(
                f"output {self.final_name}\n"
            )  # this is the name of the final file
            out.close()
        return None

    def generate_input(self) -> str:
        input_data = []
        for molecule in self.molecules:
            input_data.append(f"structure {molecule.filename}")
            input_data.append(f"  number {molecule.number_of_molecules}")
            for idx, instruction in enumerate(molecule.instructions):
                input_data.append(f"  {molecule.instructions[idx]}")
            input_data.append("end structure")

        # Convert list of input data to a single string
        return "\n".join(input_data)

    def run_packmol(self):
        validator = Validate_Fix_PDB()
        # Use the generated input to execute Packmol
        input_string = self.generate_input()
        # Write the input to a file
        with open("packmol.inp", "a") as f:
            f.write(input_string)
        # Here, run Packmol using the subprocess module or similar
        cmd = "packmol < packmol.inp"
        result = subprocess.run(cmd, shell=True, text=True, capture_output=True)
        if result.returncode != 0:
            print("Packmol failed to run with 'packmol < packmol.inp' command")
            result = subprocess.run(
                "./" + cmd, shell=True, text=True, capture_output=True
            )
            if result.returncode != 0:
                raise RuntimeError(
                    "Packmol failed to run with './packmol < packmol.inp' "
                    "command. Please check the input file and try again."
                )

        # validate final pdb
        pdb_validation = validator.validate_pdb_format(f"{self.final_name}")
        if pdb_validation[0] == 0:
            # delete .inp files
            # os.remove("packmol.inp")
            for molecule in self.molecules:
                os.remove(molecule.filename)
            # name of packed pdb file
            time_stamp = self.path_registry.get_timestamp()[-6:]
            os.rename(
                self.final_name,
                f"{self.path_registry.ckpt_pdb}/{self.final_name}",
            )
            self.path_registry.map_path(
                f"PACKED_{time_stamp}",
                f"{self.path_registry.ckpt_pdb}/{self.final_name}",
                self.file_description,
            )
            # move file to files/pdb
            print("successfull!")
            return f"PDB file validated successfully. FileID: PACKED_{time_stamp}"
        elif pdb_validation[0] == 1:
            # format pdb_validation[1] list of errors
            errors = summarize_errors(pdb_validation[1])
            # delete .inp files

            # os.remove("packmol.inp")
            print("errors:", f"{errors}")
            return "PDB file not validated, errors found {}".format(("\n").join(errors))


def packmol_wrapper(
    path_registry,
    pdbfiles: List,
    files_id: List,
    number_of_molecules: List,
    instructions: List[List],
):
    """Useful when you need to create a box
    of different types of molecules molecules"""

    # create a box
    box = PackmolBox(path_registry)
    # add molecules to the box
    for (
        pdbfile,
        file_id,
        number_of_molecules,
        instructions,
    ) in zip(pdbfiles, files_id, number_of_molecules, instructions):
        molecule = Molecule(pdbfile, file_id, number_of_molecules, instructions)
        box.add_molecule(molecule)
    # generate input header
    box.generate_input_header()
    # generate input
    # run packmol
    print("Packing:", box.file_description, "\nThe file name is:", box.final_name)
    return box.run_packmol()


"""Args schema for packmol_wrapper tool. Useful for OpenAI functions"""


class PackmolInput(BaseModel):
    pdbfiles_id: typing.Optional[typing.List[str]] = Field(
        ..., description="List of PDB files id (path_registry) to pack into a box"
    )
    small_molecules: typing.Optional[typing.List[str]] = Field(
        [],
        description=(
            "List of small molecules to be packed in the system. "
            "Examples: water, benzene, toluene, etc."
        ),
    )

    number_of_molecules: typing.Optional[typing.List[int]] = Field(
        ...,
        description=(
            "List of number of instances of each species to pack into the box. "
            "One number per species (either protein or small molecule) "
        ),
    )
    instructions: typing.Optional[typing.List[List[str]]] = Field(
        ...,
        description=(
            "List of instructions for each species. "
            "One List per Molecule. "
            "Every instruction should be one string like:\n"
            "'inside box 0. 0. 0. 90. 90. 90.'"
        ),
    )


class PackMolTool(BaseTool):
    name: str = "packmol_tool"
    description: str = (
        "Useful when you need to create a box "
        "of different types of chemical species.\n"
        "Three different examples:\n"
        "pdbfiles_id: ['1a2b_123456']\n"
        "small_molecules: ['water'] \n"
        "number_of_molecules: [1, 1000]\n"
        "instructions: [['fixed 0. 0. 0. 0. 0. 0. \n centerofmass'], "
        "['inside box 0. 0. 0. 90. 90. 90.']]\n"
        "will pack 1 molecule of 1a2b_123456 at the origin "
        "and 1000 molecules of water. \n"
        "pdbfiles_id: ['1a2b_123456']\n"
        "number_of_molecules: [1]\n"
        "instructions: [['fixed  0. 0. 0. 0. 0. 0.' \n center]]\n"
        "This will fix the barocenter of protein 1a2b_123456 at "
        "the center of the box with no rotation.\n"
        "pdbfiles_id: ['1a2b_123456']\n"
        "number_of_molecules: [1]\n"
        "instructions: [['outside sphere 2.30 3.40 4.50 8.0]]\n"
        "This will place the protein 1a2b_123456 outside a sphere "
        "centered at 2.30 3.40 4.50 with radius 8.0\n"
    )

    args_schema: Type[BaseModel] = PackmolInput

    path_registry: typing.Optional[PathRegistry]

    def __init__(self, path_registry: typing.Optional[PathRegistry]):
        super().__init__()
        self.path_registry = path_registry

    def _get_sm_pdbs(self, small_molecules):
        all_files = self.path_registry.list_path_names()
        for molecule in small_molecules:
            # check path registry for molecule.pdb
            if molecule not in all_files:
                # download molecule using small_molecule_pdb from MolPDB
                molpdb = MolPDB(self.path_registry)
                molpdb.small_molecule_pdb(molecule)
        print("Small molecules PDBs created successfully")

    @validate_tool_args(args_schema=args_schema)
    def _run(self, **values) -> str:
        """use the tool."""

        if self.path_registry is None:  # this should not happen
            raise ValidationError("Path registry not initialized")
        try:
            values = self.validate_input(values)
        except ValidationError as e:
            return str(e)
        error_msg = values.get("error", None)
        if error_msg:
            print("Error in Packmol inputs:", error_msg)
            return f"Error in inputs: {error_msg}"
        print("Starting Packmol Tool!")
        pdbfile_ids = values.get("pdbfiles_id", [])
        pdbfiles = [
            self.path_registry.get_mapped_path(pdbfile) for pdbfile in pdbfile_ids
        ]
        pdbfile_names = [pdbfile.split("/")[-1] for pdbfile in pdbfiles]
        # copy them to the current directory with temp_ names

        pdbfile_names = [
            f"{self.path_registry.ckpt_files}/temp_{pdbfile_name}"
            for pdbfile_name in pdbfile_names
        ]
        number_of_molecules = values.get("number_of_molecules", [])
        instructions = values.get("instructions", [])
        small_molecules = values.get("small_molecules", [])
        # make sure small molecules are all downloaded
        self._get_sm_pdbs(small_molecules)
        small_molecules_files = [
            self.path_registry.get_mapped_path(sm) for sm in small_molecules
        ]
        small_molecules_file_names = [
            small_molecule.split("/")[-1] for small_molecule in small_molecules_files
        ]
        small_molecules_file_names = [
            f"{self.path_registry.ckpt_files}/temp_{small_molecule_file_name}"
            for small_molecule_file_name in small_molecules_file_names
        ]
        # append small molecules to pdbfiles
        pdbfiles.extend(small_molecules_files)
        pdbfile_names.extend(small_molecules_file_names)
        pdbfile_ids.extend(small_molecules)

        for pdbfile, pdbfile_name in zip(pdbfiles, pdbfile_names):
            os.system(f"cp {pdbfile} {pdbfile_name}")
        # check if packmol is installed
        cmd = "command -v packmol"
        result = subprocess.run(cmd, shell=True, text=True, capture_output=True)
        if result.returncode != 0:
            result = subprocess.run(
                "./" + cmd, shell=True, text=True, capture_output=True
            )
            if result.returncode != 0:
                return (
                    "Packmol is not installed. Please install"
                    "packmol at "
                    "'https://m3g.github.io/packmol/download.shtml'"
                    "and try again."
                )
        try:
            return packmol_wrapper(
                self.path_registry,
                pdbfiles=pdbfile_names,
                files_id=pdbfile_ids,
                number_of_molecules=number_of_molecules,
                instructions=instructions,
            )
        except RuntimeError as e:
            return f"Packmol failed to run with error: {e}"

    def validate_input(cls, values: Union[str, Dict[str, Any]]) -> Dict:
        # check if is only a string
        if isinstance(values, str):
            print("values is a string", values)
            raise ValidationError("Input must be a dictionary")
        pdbfiles = values.get("pdbfiles_id", [])
        small_molecules = values.get("small_molecules", [])
        number_of_molecules = values.get("number_of_molecules", [])
        instructions = values.get("instructions", [])
        number_of_species = len(pdbfiles) + len(small_molecules)

        if not number_of_species == len(number_of_molecules):
            if not number_of_species == len(instructions):
                return {
                    "error": (
                        "The length of number_of_molecules AND instructions "
                        "must be equal to the number of species in the system. "
                        f"You have {number_of_species} "
                        f"from {len(pdbfiles)} pdbfiles and {len(small_molecules)} "
                        "small molecules. You have included "
                        f"{len(number_of_molecules)} values for "
                        f"number_of_molecules and {len(instructions)}"
                        "instructions."
                    )
                }
            return {
                "error": (
                    "The length of number_of_molecules must be equal to the "
                    f"number of species in the system. You have {number_of_species} "
                    f"from {len(pdbfiles)} pdbfiles and {len(small_molecules)} "
                    f"small molecules. You have included "
                    f"{len(number_of_molecules)} values "
                    "for number_of_molecules"
                )
            }
        elif not number_of_species == len(instructions):
            return {
                "error": (
                    "The length of instructions must be equal to the "
                    f"number of species in the system. You have {number_of_species} "
                    f"from {len(pdbfiles)} pdbfiles and {len(small_molecules)} "
                    "small molecules. You have included "
                    f"{len(instructions)} instructions."
                )
            }
        registry = PathRegistry.get_instance()
        molPDB = MolPDB(registry)
        for instruction in instructions:
            if len(instruction) != 1:
                return {
                    "error": (
                        "Each instruction must be a single string. "
                        "If necessary, use newlines in a instruction string."
                    )
                }
            # TODO enhance this validation with more packmol instructions
            first_word = instruction[0].split(" ")[0]
            if first_word == "center":
                if len(instruction[0].split(" ")) == 1:
                    return {
                        "error": (
                            "The instruction 'center' must be "
                            "accompanied by more instructions. "
                            "Example 'fixed 0. 0. 0. 0. 0. 0.' "
                            "The complete instruction would be: 'center \n fixed 0. 0. "
                            "0. 0. 0. 0.' with a newline separating the two "
                            "instructions."
                        )
                    }
            elif first_word not in [
                "inside",
                "outside",
                "fixed",
            ]:
                return {
                    "error": (
                        "The first word of each instruction must be one of "
                        "'inside' or 'outside' or 'fixed' \n"
                        "examples: center \n fixed 0. 0. 0. 0. 0. 0.,\n"
                        "inside box -10. 0. 0. 10. 10. 10. \n"
                    )
                }

        # Further validation, e.g., checking if files exist
        file_ids = registry.list_path_names()

        for pdbfile_id in pdbfiles:
            if "_" not in pdbfile_id:
                return {
                    "error": (
                        f"{pdbfile_id} is not a valid pdbfile_id in the path_registry"
                    )
                }
            if pdbfile_id not in file_ids:
                # look for files in the current directory
                # that match some part of the pdbfile
                ids_w_description = registry.list_path_names_and_descriptions()

                return {
                    "error": (
                        f"PDB file ID {pdbfile_id} does not exist "
                        "in the path registry.\n"
                        f"This are the files IDs: {ids_w_description} "
                    )
                }
            for small_molecule in small_molecules:
                if small_molecule not in file_ids:
                    result = molPDB.small_molecule_pdb(small_molecule)
                    if "successfully" not in result:
                        return {
                            "error": (
                                f"{small_molecule} could not be converted to a pdb "
                                "file. Try with a different name, or with the SMILES "
                                "of the small molecule"
                            )
                        }
        return values

    async def _arun(self, values: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")
