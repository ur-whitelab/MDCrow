import os
import re
import sys
import typing
from typing import Dict, Optional, Type

from langchain.tools import BaseTool
from pdbfixer import PDBFixer
from pydantic import BaseModel, Field, ValidationError

from mdcrow.utils import PathRegistry

from .elements import list_of_elements


class PDBsummarizerfxns:
    def __init__(self):
        self.list_of_elements = list_of_elements

    def _record_inf(self, pdbfile):
        with open(pdbfile, "r") as f:
            lines = f.readlines()
            remarks = [
                record_lines
                for record_lines in lines
                if record_lines.startswith("REMARK")
            ]
            atoms = [
                record_lines
                for record_lines in lines
                if record_lines.startswith("ATOM")
            ]
            box = [
                record_lines
                for record_lines in lines
                if record_lines.startswith("CRYST1")
            ]
            HETATM = [
                record_lines
                for record_lines in lines
                if record_lines.startswith("HETATM")
            ]

        return remarks, atoms, box, HETATM

    def _num_of_dif_residues(self, pdbfile):
        remarks, atoms, box, HETATM = self._record_inf(pdbfile)
        residues = [atom[17:20] for atom in atoms]
        residues = list(set(residues))
        return len(residues)

    # diagnosis
    """Checks for the elements names in the pdb file.
    Positions 76-78 of the ATOM and HETATM records"""

    def _atoms_have_elements(self, pdbfile):
        _, atoms, _, _ = self._record_inf(pdbfile)
        print(atoms)
        elements = [atom[76:78] for atom in atoms if atom not in [" ", "", "  ", "   "]]
        print(elements)
        if len(elements) != len(atoms):
            print(
                (
                    "No elements in the ATOM records there are"
                    "{len(elements)} elements and {len(atoms)}"
                    "atoms records"
                )
            )
            return False
        elements = list(set(elements))
        for element in elements:
            if element not in self.list_of_elements:
                print("Element not in the list of elements")
                return False
        return True

    def _atoms_have_tempFactor(self, pdbfile):
        _, atoms, _, _ = self._record_inf(pdbfile)
        tempFactor = [
            atom[60:66]
            for atom in atoms
            if atom[60:66] not in [" ", "", "  ", "   ", "  ", "      "]
        ]
        if len(tempFactor) != len(atoms):
            return False
        return True

    def _atoms_have_occupancy(self, pdbfile):
        _, atoms, _, _ = self._record_inf(pdbfile)
        occupancy = [
            atom[54:60]
            for atom in atoms
            if atom[54:60] not in [" ", "", "  ", "   ", "  ", "      "]
        ]
        if len(occupancy) != len(atoms):
            return False
        return True

    def _hetatom_have_occupancy(self, pdbfile):
        _, _, _, HETATM = self._record_inf(pdbfile)
        occupancy = [
            atom[54:60]
            for atom in HETATM
            if atom[54:60] not in [" ", "", "  ", "   ", "  ", "      "]
        ]
        if len(occupancy) != len(HETATM):
            return False
        return True

    def _hetatm_have_elements(self, pdbfile):
        _, _, _, HETATM = self._record_inf(pdbfile)
        elements = [
            atom[76:78] for atom in HETATM if atom[76:78] not in [" ", "", "  ", "   "]
        ]
        if len(elements) != len(HETATM):
            print("No elements in the HETATM records")
            return False
        return True

    def _hetatm_have_tempFactor(self, pdbfile):
        _, _, _, HETATM = self._record_inf(pdbfile)
        tempFactor = [
            atom[60:66] for atom in HETATM if atom not in [" ", "", "  ", "   "]
        ]
        if len(tempFactor) != len(HETATM):
            return False
        return True

    """Checks for the residue names in the pdb file.
      Positions 17-20 of the ATOM and HETATM records"""

    def _atoms_hetatm_have_residue_names(self, pdbfile):
        _, atoms, _, HETATM = self._record_inf(pdbfile)
        residues = [atom[17:20] for atom in atoms]
        residues = list(set(residues))
        if len(residues) != len(atoms):
            return False
        residues = [atom[17:20] for atom in HETATM]
        residues = list(set(residues))
        if len(residues) != len(HETATM):
            return False
        return True

    def _atoms_hetatm_have_occupancy(self, pdbfile):
        _, atoms, _, HETATM = self._record_inf(pdbfile)
        occupancy = [
            atom[54:60]
            for atom in atoms
            if atom not in [" ", "", "  ", "   ", "  ", "      "]
        ]
        if len(occupancy) != len(atoms):
            return False
        occupancy = [
            HET[54:60]
            for HET in HETATM
            if HET not in [" ", "", "  ", "   ", "  ", "      "]
        ]
        if len(occupancy) != len(HETATM):
            return False
        return True

    def _non_standard_residues(self, pdbfile):
        fixer = PDBFixer(file_name=pdbfile)
        fixer.findNonstandardResidues()
        len(fixer.nonstandardResidues)

    def pdb_summarizer(self, pdb_file):
        self.remarks, self.atoms, self.box, self.HETATM = self._record_inf(pdb_file)
        self.atoms_elems = self._atoms_have_elements(pdb_file)
        self.HETATM_elems = self._hetatm_have_elements(pdb_file)
        self.residues = self._atoms_hetatm_have_residue_names(pdb_file)
        self.atoms_tempFact = self._atoms_have_tempFactor(pdb_file)
        self.num_of_residues = self._num_of_dif_residues(pdb_file)
        self.HETATM_tempFact = self._hetatm_have_tempFactor(pdb_file)

        output = (
            f"PDB file: {pdb_file} has the following properties:"
            "Number of residues: {pdb.num_of_residues}"
            "Are elements identifiers present: {pdb.atoms}"
            "Are HETATM elements identifiers present: {pdb.HETATM}"
            "Are residue names present: {pdb.residues}"
            "Are box dimensions present: {pdb.box}"
            "Non-standard residues: {pdb.HETATM}"
        )
        return output


class Validate_Fix_PDB:
    def validate_pdb_format(self, fhandle):
        """
        Compare each ATOM/HETATM line with the format defined on the
        official PDB website.

        Parameters
        ----------
        fhandle : a line-by-line iterator of the original PDB file.

        Returns
        -------
        (int, list)
            - 1 if error was found, 0 if no errors were found.
            - List of error messages encountered.
        """
        # check if filename is in directory
        if not os.path.exists(fhandle):
            return (1, ["File not found. Packmol failed to write the file."])
        errors = []
        _fmt_check = (
            ("Atm. Num.", (slice(6, 11), re.compile(r"[\d\s]+"))),
            ("Alt. Loc.", (slice(11, 12), re.compile(r"\s"))),
            ("Atm. Nam.", (slice(12, 16), re.compile(r"\s*[A-Z0-9]+\s*"))),
            ("Spacer #1", (slice(16, 17), re.compile(r"[A-Z0-9 ]{1}"))),
            ("Res. Nam.", (slice(17, 20), re.compile(r"\s*[A-Z0-9]+\s*"))),
            ("Spacer #2", (slice(20, 21), re.compile(r"\s"))),
            ("Chain Id.", (slice(21, 22), re.compile(r"[A-Za-z0-9 ]{1}"))),
            ("Res. Num.", (slice(22, 26), re.compile(r"\s*[\d\-]+\s*"))),
            ("Ins. Code", (slice(26, 27), re.compile(r"[A-Z0-9 ]{1}"))),
            ("Spacer #3", (slice(27, 30), re.compile(r"\s+"))),
            ("Coordn. X", (slice(30, 38), re.compile(r"\s*[\d\.\-]+\s*"))),
            ("Coordn. Y", (slice(38, 46), re.compile(r"\s*[\d\.\-]+\s*"))),
            ("Coordn. Z", (slice(46, 54), re.compile(r"\s*[\d\.\-]+\s*"))),
            ("Occupancy", (slice(54, 60), re.compile(r"\s*[\d\.\-]+\s*"))),
            ("Tmp. Fac.", (slice(60, 66), re.compile(r"\s*[\d\.\-]+\s*"))),
            ("Spacer #4", (slice(66, 72), re.compile(r"\s+"))),
            ("Segm. Id.", (slice(72, 76), re.compile(r"[\sA-Z0-9\-\+]+"))),
            ("At. Elemt", (slice(76, 78), re.compile(r"[\sA-Z0-9\-\+]+"))),
            ("At. Charg", (slice(78, 80), re.compile(r"[\sA-Z0-9\-\+]+"))),
        )

        def _make_pointer(column):
            col_bg, col_en = column.start, column.stop
            pt = ["^" if c in range(col_bg, col_en) else " " for c in range(80)]
            return "".join(pt)

        for iline, line in enumerate(fhandle, start=1):
            line = line.rstrip("\n").rstrip("\r")  # CR/LF
            if not line:
                continue

            if line[0:6] in ["ATOM  ", "HETATM"]:
                # ... [rest of the code unchanged here]
                linelen = len(line)
                if linelen < 80:
                    emsg = "[!] Line {0} is short: {1} < 80\n"
                    sys.stdout.write(emsg.format(iline, linelen))

                elif linelen > 80:
                    emsg = "[!] Line {0} is long: {1} > 80\n"
                    sys.stdout.write(emsg.format(iline, linelen))

                for fname, (fcol, fcheck) in _fmt_check:
                    field = line[fcol]
                    if not fcheck.match(field):
                        pointer = _make_pointer(fcol)
                        emsg = "[!] Offending field ({0}) at line {1}\n".format(
                            fname, iline
                        )
                        emsg += repr(line) + "\n"
                        emsg += pointer + "\n"
                        errors.append(emsg)

            else:
                # ... [rest of the code unchanged here]
                linelen = len(line)
                # ... [rest of the code unchanged here]
                linelen = len(line)
                skip_keywords = (
                    "END",
                    "ENDMDL",
                    "HEADER",
                    "TITLE",
                    "REMARK",
                    "CRYST1",
                    "MODEL",
                )

                if any(keyword in line for keyword in skip_keywords):
                    continue

                if linelen < 80:
                    emsg = "[!] Line {0} is short: {1} < 80\n"
                    sys.stdout.write(emsg.format(iline, linelen))
                elif linelen > 80:
                    emsg = "[!] Line {0} is long: {1} > 80\n"
                    sys.stdout.write(emsg.format(iline, linelen))

        """
        map paths to files in path_registry before you return the string
        same for all other functions you want to save files for next tools
        Don't forget to import PathRegistry and add path_registry
        or PathRegistry as an argument
        """
        if errors:
            msg = "\nTo understand your errors, read the format specification:\n"
            msg += "http://www.wwpdb.org/documentation/file-format-content/format33/sect9.html#ATOM\n"
            errors.append(msg)
            return (1, errors)
        else:
            return (0, ["It *seems* everything is OK."])

    def _fix_element_column(self, pdb_file, custom_element_dict=None):
        records = ("ATOM", "HETATM", "ANISOU")
        corrected_lines = []
        for line in pdb_file:
            if line.startswith(records):
                atom_name = line[12:16]

                if atom_name[0].isalpha() and not atom_name[2:].isdigit():
                    element = atom_name.strip()
                else:
                    atom_name = atom_name.strip()
                    if atom_name[0].isdigit():
                        element = atom_name[1]
                    else:
                        element = atom_name[0]

                if element not in set(list_of_elements):
                    element = "  "  # empty element in case we cannot assign

                line = line[:76] + element.rjust(2) + line[78:]
                corrected_lines.append(line)

        return corrected_lines

    def fix_element_column(self, pdb_file, custom_element_dict=None):
        """Fixes the Element columns of a pdb file"""

        # extract Title, Header, Remarks, and Cryst1 records
        file_name = pdb_file.split(".")[0]
        # check if theres a file-name-fixed.pdb file
        if os.path.isfile(file_name + "-fixed.pdb"):
            pdb_file = file_name + "-fixed.pdb"
        assert isinstance(pdb_file, str), "pdb_file must be a string"
        with open(pdb_file, "r") as f:
            print("I read the initial file")
            pdb_file_lines = f.readlines()
            # validate if pdbfile has element records
            pdb = PDBsummarizerfxns()
            atoms_have_elems, HETATM_have_elems = pdb._atoms_have_elements(
                pdb_file
            ), pdb._hetatm_have_elements(pdb_file)
            if atoms_have_elems and HETATM_have_elems:
                f.close()
                return (
                    "Element's column already filled with"
                    "elements, no fix needed for elements"
                )
            print("I closed the initial file")
            f.close()

        # fix element column
        records = ("TITLE", "HEADER", "REMARK", "CRYST1", "HET", "LINK", "SEQRES")
        final_records = ("CONECT", "MASTER", "END")
        _unchanged_records = []
        _unchanged_final_records = []
        print("pdb_file", pdb_file)
        for line in pdb_file_lines:
            if line.startswith(records):
                _unchanged_records.append(line)
            elif line.startswith(final_records):
                _unchanged_final_records.append(line)
        print("_unchanged_records", _unchanged_records)
        new_pdb = self._fix_element_column(pdb_file_lines, custom_element_dict)
        # join the linees
        new_pdb = "".join(new_pdb)
        # write new pdb file as pdb_file-fixed.pdb
        new_pdb_file = file_name.split(".")[0] + "-fixed.pdb"
        print("name of fixed pdb file", new_pdb_file)
        # write the unchanged records first and then the new pdb file
        assert isinstance(new_pdb_file, str), "new_pdb_file must be a string"
        with open(new_pdb_file, "w") as f:
            print("I wrote the new file")
            f.writelines(_unchanged_records)
            f.write(new_pdb)
            f.writelines(_unchanged_final_records)
            f.close()
        try:
            # read the new pdb file and check if it has element records
            with open(new_pdb_file, "r") as f:
                pdb_file_lines = f.readlines()
                pdb = PDBsummarizerfxns()
                atoms_have_elems, HETATM_have_elems = pdb._atoms_have_elements(
                    new_pdb_file
                ), pdb._hetatm_have_elements(new_pdb_file)
                if atoms_have_elems and HETATM_have_elems:
                    f.close()
                    return "Element's column fixed successfully"
                else:
                    f.close()
                    return "Element's column not fixed, and i dont know why"
        except Exception as e:
            return f"Element's column not fixed error: {e}"

    def pad_line(self, line):
        """Pad line to 80 characters in case it is shorter."""
        size_of_line = len(line)
        if size_of_line < 80:
            padding = 80 - size_of_line + 1
            line = line.strip("\n") + " " * padding + "\n"
        return line[:81]  # 80 + newline character

    def _fix_temp_factor_column(self, pdbfile, bfactor, only_fill):
        """Set the temperature column in all ATOM/HETATM records to a given value.

        This function is a generator.

        Parameters
        ----------
        fhandle : a line-by-line iterator of the original PDB file.

        bfactor : float
            The desired bfactor.

        Yields
        ------
        str (line-by-line)
            The modified (or not) PDB line."""
        _pad_line = self.pad_line
        records = ("ATOM", "HETATM")
        corrected_lines = []
        bfactor = "{0:>6.2f}".format(bfactor)

        for line in pdbfile:
            if line.startswith(records):
                line = _pad_line(line)
                if only_fill:
                    if line[60:66].strip() == "":
                        corrected_lines.append(line[:60] + bfactor + line[66:])
                else:
                    corrected_lines.append(line[:60] + bfactor + line[66:])
            else:
                corrected_lines.append(line)

        return corrected_lines

    def fix_temp_factor_column(self, pdb_file, bfactor=1.00, only_fill=True):
        """Fixes the tempFactor columns of a pdb file"""

        # extract Title, Header, Remarks, and Cryst1 records
        # get name from pdb_file
        if isinstance(pdb_file, str):
            file_name = pdb_file.split(".")[0]
        else:
            return "pdb_file must be a string"
        file_name = pdb_file.split(".")[0]

        if os.path.isfile(file_name + "-fixed.pdb"):
            file_name = file_name + "-fixed.pdb"

        assert isinstance(file_name, str), "pdb_file must be a string"
        with open(file_name, "r") as f:
            print("im reading the files temp factor")
            pdb_file_lines = f.readlines()
            # validate if pdbfile has temp factors
            pdb = PDBsummarizerfxns()
            atoms_have_bfactor, HETATM_have_bfactor = pdb._atoms_have_tempFactor(
                pdb_file
            ), pdb._hetatm_have_tempFactor(pdb_file)
            if atoms_have_bfactor and HETATM_have_bfactor and only_fill:
                # print("Im closing the file temp factor")
                f.close()
                return (
                    "TempFact column filled with bfactor already,"
                    "no fix needed for temp factor"
                )
            f.close()
        # fix element column
        records = ("TITLE", "HEADER", "REMARK", "CRYST1", "HET", "LINK", "SEQRES")
        final_records = ("CONECT", "MASTER", "END")
        _unchanged_final_records = []
        _unchanged_records = []
        for line in pdb_file_lines:
            if line.startswith(records):
                _unchanged_records.append(line)
            elif line.startswith(final_records):
                _unchanged_final_records.append(line)

        new_pdb = self._fix_temp_factor_column(pdb_file_lines, bfactor, only_fill)
        # join the linees
        new_pdb = "".join(new_pdb)
        # write new pdb file as pdb_file-fixed.pdb
        new_pdb_file = file_name + "-fixed.pdb"
        # organize columns:
        # HEADER, TITLE, REMARKS, CRYST1, ATOM, HETATM, CONECT, MASTER, END

        assert isinstance(new_pdb_file, str), "new_pdb_file must be a string"
        # write new pdb file as pdb_file-fixed.pdb
        with open(new_pdb_file, "w") as f:
            f.writelines(_unchanged_records)
            f.write(new_pdb)
            f.writelines(_unchanged_final_records)
            f.close()
        try:
            # read the new pdb file and check if it has element records
            with open(new_pdb_file, "r") as f:
                pdb_file = f.readlines()
                pdb = PDBsummarizerfxns()
                atoms_have_bfactor, HETATM_have_bfactor = pdb._atoms_have_tempFactor(
                    new_pdb_file
                ), pdb._hetatm_have_tempFactor(new_pdb_file)
                if atoms_have_bfactor and HETATM_have_bfactor:
                    f.close()
                    return "TempFact fixed successfully"
                else:
                    f.close()
                    return "TempFact column not fixed"
        except Exception as e:
            return f"Couldnt read written file TempFact column not fixed error: {e}"

    def _fix_occupancy_column(self, pdbfile, occupancy, only_fill):
        """
        Set the occupancy column in all ATOM/HETATM records to a given value.

        Non-ATOM/HETATM lines are give as are. This function is a generator.

        Parameters
        ----------
        fhandle : a line-by-line iterator of the original PDB file.

        occupancy : float
            The desired occupancy value

        Yields
        ------
        str (line-by-line)
            The modified (or not) PDB line.
        """

        records = ("ATOM", "HETATM")
        corrected_lines = []
        occupancy = "{0:>6.2f}".format(occupancy)
        for line in pdbfile:
            if line.startswith(records):
                line = self.pad_line(line)
                if only_fill:
                    if line[54:60].strip() == "":
                        corrected_lines.append(line[:54] + occupancy + line[60:])
                else:
                    corrected_lines.append(line[:54] + occupancy + line[60:])
            else:
                corrected_lines.append(line)

        return corrected_lines

    def fix_occupancy_columns(self, pdb_file, occupancy=1.0, only_fill=True):
        """Fixes the occupancy columns of a pdb file"""
        # extract Title, Header, Remarks, and Cryst1 records
        # get name from pdb_file
        file_name = pdb_file.split(".")[0]
        if os.path.isfile(file_name + "-fixed.pdb"):
            file_name = file_name + "-fixed.pdb"

        assert isinstance(pdb_file, str), "pdb_file must be a string"
        with open(file_name, "r") as f:
            pdb_file_lines = f.readlines()
            # validate if pdbfile has occupancy
            pdb = PDBsummarizerfxns()
            atoms_have_bfactor, HETATM_have_bfactor = pdb._atoms_have_occupancy(
                file_name
            ), pdb._hetatom_have_occupancy(file_name)
            if atoms_have_bfactor and HETATM_have_bfactor and only_fill:
                f.close()
                return (
                    "Occupancy column filled with occupancy"
                    "already, no fix needed for occupancy"
                )
            f.close()
        # fix element column
        records = ("TITLE", "HEADER", "REMARK", "CRYST1", "HET", "LINK", "SEQRES")
        final_records = ("CONECT", "MASTER", "END")
        _unchanged_records = []
        _unchanged_final_records = []
        for line in pdb_file_lines:
            if line.startswith(records):
                _unchanged_records.append(line)
            elif line.startswith(final_records):
                _unchanged_final_records.append(line)

        new_pdb = self._fix_occupancy_column(pdb_file_lines, occupancy, only_fill)
        # join the linees
        new_pdb = "".join(new_pdb)
        # write new pdb file as pdb_file-fixed.pdb
        new_pdb_file = file_name + "-fixed.pdb"

        # write new pdb file as pdb_file-fixed.pdb
        assert isinstance(new_pdb_file, str), "new_pdb_file must be a string"
        with open(new_pdb_file, "w") as f:
            f.writelines(_unchanged_records)
            f.write(new_pdb)
            f.writelines(_unchanged_final_records)
            f.close()
        try:
            # read the new pdb file and check if it has element records
            with open(new_pdb_file, "r") as f:
                pdb_file = f.readlines()
                pdb = PDBsummarizerfxns()
                atoms_have_bfactor, HETATM_have_bfactor = pdb._atoms_have_tempFactor(
                    new_pdb_file
                ), pdb._hetatm_have_tempFactor(new_pdb_file)
                if atoms_have_bfactor and HETATM_have_bfactor:
                    f.close()
                    return "Occupancy fixed successfully"
                else:
                    f.close()
                    return "Occupancy column not fixed"
        except Exception:
            return "Couldnt read file Occupancy's column not fixed"

    def apply_fixes(self, pdbfile, query):
        # Define a mapping between query keys and functions.
        # If a function requires additional arguments from the query,
        # define it as a lambda.
        FUNCTION_MAP = {
            "ElemColum": lambda pdbfile, params: self.fix_element_column(pdbfile),
            "tempFactor": lambda pdbfile, params: self.fix_temp_factor_column(
                pdbfile, *params
            ),
            "Occupancy": lambda pdbfile, params: self.fix_occupancy_columns(
                pdbfile, *params
            ),
        }
        # Iterate through the keys and functions in FUNCTION_MAP.
        for key, func in FUNCTION_MAP.items():
            # Check if the current key is in the query and is not None.
            params = query.get(key)
            if params is not None:
                # If it is, call the function with
                # pdbfile and the parameters from the query.
                func(pdbfile, params)

                return "PDB file fixed"


class PDBFilesFixInp(BaseModel):
    pdbfile: str = Field(..., description="PDB file to be fixed")
    ElemColum: typing.Optional[bool] = Field(
        False,
        description=(
            "List of fixes to be applied. If None, a"
            "validation of what fixes are needed is performed."
        ),
    )
    tempFactor: typing.Optional[typing.Tuple[float, bool]] = Field(
        (...),
        description=(
            "Tuple of     ( float, bool)"
            "first arg is the"
            "value to be set as the tempFill, and third arg indicates"
            "if only empty TempFactor columns have to be filled"
        ),
    )
    Occupancy: typing.Optional[typing.Tuple[float, bool]] = Field(
        (...),
        description=(
            "Tuple of (bool, float, bool)"
            "where first arg indicates if Occupancy"
            "fix has to be applied, second arg is the"
            "value to be set, and third arg indicates"
            "if only empty Occupancy columns have to be filled"
        ),
    )


class FixPDBFile(BaseTool):
    name: str = "PDBFileFixer"
    description: str = "Fixes PDB files columns if needed"
    args_schema: Type[BaseModel] = PDBFilesFixInp

    path_registry: Optional[PathRegistry]

    def __init__(self, path_registry: Optional[PathRegistry]):
        super().__init__()
        self.path_registry = path_registry

    def _run(self, query: Dict):
        """use the tool."""
        if self.path_registry is None:
            raise ValidationError("Path registry not initialized")
        pdb_ff = Validate_Fix_PDB()
        error_msg = query.get("error")
        if error_msg:
            return "Failed. " + error_msg
        pdbfile = query.pop("pdbfile")
        if len(query.keys()) == 0:
            validation = pdb_ff.validate_pdb_format(pdbfile)
            if validation[0] == 0:
                return "Succeeded. PDB file is valid, no need to fix it"

            if validation[0] == 1:
                # Convert summarized_errors into a set for O(1) lookups
                error_set = set(validation[1])

                # Apply Fixes
                if "At. Elem." in error_set:
                    pdb_ff.fix_element_column(pdbfile)
                if "Tmp. Fac." in error_set:
                    pdb_ff.fix_temp_factor_column(pdbfile)
                if "Occupancy" in error_set:
                    pdb_ff.fix_occupancy_columns(pdbfile)

                validate = pdb_ff.validate_pdb_format(pdbfile + "-fixed.pdb")
                if validate[0] == 0:
                    name = pdbfile + "-fixed.pdb"
                    description = "PDB file fixed"
                    self.path_registry.map_path(name, name, description)
                    return "Succeeded. PDB file fixed"
                else:
                    return "Failed. PDB not fully fixed"
        else:
            pdb_ff.apply_fixes(pdbfile, query)
            validate = pdb_ff.validate_pdb_format(pdbfile + "-fixed.pdb")
            if validate[0] == 0:
                name = pdbfile + "-fixed.pdb"
                description = "PDB file fixed"
                self.path_registry.map_path(name, name, description)
                return "Succeeded. PDB file fixed"
            else:
                return "Failed. PDB not fully fixed"
