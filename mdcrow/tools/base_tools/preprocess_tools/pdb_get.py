from typing import Optional

import requests
from langchain.tools import BaseTool
from rdkit import Chem
from rdkit.Chem import AllChem

from mdcrow.utils import FileType, PathRegistry


def get_pdb(query_string: str, path_registry: PathRegistry):
    """
    Search RSCB's protein data bank using the given query string
    and return the path to pdb file in either CIF or PDB format
    """
    if path_registry is None:
        path_registry = PathRegistry.get_instance()
    url = "https://search.rcsb.org/rcsbsearch/v2/query?json={search-request}"
    query = {
        "query": {
            "type": "terminal",
            "service": "full_text",
            "parameters": {"value": query_string},
        },
        "return_type": "entry",
    }
    r = requests.post(url, json=query)
    if r.status_code == 204:
        return None, None
    if "cif" in query_string or "CIF" in query_string:
        filetype = "cif"
    else:
        filetype = "pdb"
    if "result_set" in r.json() and len(r.json()["result_set"]) > 0:
        results = r.json()["result_set"]
        pdbid = max(results, key=lambda x: x["score"])["identifier"]
        print(f"PDB file found with this ID: {pdbid}")
        url = f"https://files.rcsb.org/download/{pdbid}.{filetype}"
        pdb = requests.get(url)
        filename = path_registry.write_file_name(
            FileType.PROTEIN,
            protein_name=pdbid,
            description="raw",
            file_format=filetype,
        )
        file_id = path_registry.get_fileid(filename, FileType.PROTEIN)
        directory = f"{path_registry.ckpt_pdb}"

        with open(f"{directory}/{filename}", "w") as file:
            file.write(pdb.text)
        path_registry.map_path(
            file_id,
            f"{path_registry.ckpt_pdb}/{filename}",
            "PDB file downloaded from RSCB",
        )

        return filename, file_id
    return None, None


class ProteinName2PDBTool(BaseTool):
    name = "PDBFileDownloader"
    description = (
        "This tool downloads PDB (Protein Data Bank) or"
        "CIF (Crystallographic Information File) files using"
        "a protein's common name (NOT a small molecule)."
        "When a specific file type, either PDB or CIF,"
        "is requested, add file type to the query string with space."
        "Input: Commercial name of the protein or file without"
        "file extension"
        "Output: Corresponding PDB or CIF file"
    )
    path_registry: Optional[PathRegistry]

    def __init__(self, path_registry: Optional[PathRegistry]):
        super().__init__()
        self.path_registry = path_registry

    def _run(self, query: str) -> str:
        """Use the tool."""
        try:
            if self.path_registry is None:  # this should not happen
                return "Failed. Path registry not initialized"
            filename, pdbfile_id = get_pdb(query, self.path_registry)
            if pdbfile_id is None:
                return "Failed. Name2PDB tool failed to find and download PDB file."
            else:
                self.path_registry.map_path(
                    pdbfile_id,
                    f"{self.path_registry.ckpt_pdb}/{filename}",
                    f"PDB file downloaded from RSCB, PDBFile ID: {pdbfile_id}",
                )
                return f"Succeeded. Downloaded the PDB file:{pdbfile_id}"
        except Exception as e:
            return f"Failed. {type(e).__name__}: {str(e)}"

    async def _arun(self, query) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("this tool does not support async")


class MolPDB:
    def __init__(self, path_registry):
        self.path_registry = path_registry

    def is_smiles(self, text: str) -> bool:
        try:
            m = Chem.MolFromSmiles(text, sanitize=False)
            if m is None:
                return False
            return True
        except Exception:
            return False

    def largest_mol(
        self, smiles: str
    ) -> (
        str
    ):  # from https://github.com/ur-whitelab/chemcrow-public/blob/main/chemcrow/utils.py
        ss = smiles.split(".")
        ss.sort(key=lambda a: len(a))
        while not self.is_smiles(ss[-1]):
            rm = ss[-1]
            ss.remove(rm)
        return ss[-1]

    def molname2smiles(
        self, query: str
    ) -> (
        str
    ):  # from https://github.com/ur-whitelab/chemcrow-public/blob/main/chemcrow/tools/databases.py
        url = " https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{}/{}"
        r = requests.get(url.format(query, "property/IsomericSMILES/JSON"))
        # convert the response to a json object
        data = r.json()
        # return the SMILES string
        try:
            smi = data["PropertyTable"]["Properties"][0]["IsomericSMILES"]
        except KeyError:
            return (
                "Could not find a molecule matching the text."
                "One possible cause is that the input is incorrect, "
                "input one molecule at a time."
            )
        # remove salts
        return Chem.CanonSmiles(self.largest_mol(smi))

    def smiles2name(self, smi: str) -> str:
        try:
            smi = Chem.MolToSmiles(Chem.MolFromSmiles(smi), canonical=True)
        except Exception:
            return "Invalid SMILES string"
        # query the PubChem database
        r = requests.get(
            "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/"
            + smi
            + "/synonyms/JSON"
        )
        data = r.json()
        try:
            name = data["InformationList"]["Information"][0]["Synonym"][0]
        except KeyError:
            return "Unknown Molecule"
        return name

    def get_pubchem_cid(self, smiles):
        _url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/"
        url = _url + f"{smiles}/cids/JSON"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            return data["IdentifierList"]["CID"][0]
        return None

    def get_hetcode_from_cid(self, cid):
        print(cid)
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{cid}/JSON"
        response = requests.get(url)
        header_1 = "Interactions and Pathways"
        header_2 = "Protein Bound 3D Structures"
        header_3 = "Ligands from Protein Bound 3D Structures"
        header_4 = "PDBe Ligand Code"
        if response.status_code == 200:
            data = response.json()
            for section in data["Record"]["Section"]:
                if section["TOCHeading"] == header_1:
                    for subsection in section["Section"]:
                        if subsection["TOCHeading"] == header_2:
                            for subsubsection in subsection["Section"]:
                                if subsubsection["TOCHeading"] == header_3:
                                    for s in subsubsection["Section"]:
                                        if s["TOCHeading"] == header_4:
                                            return s["Information"][0]["Value"][
                                                "StringWithMarkup"
                                            ][0]["String"]
        return None

    def get_hetcode(self, smiles):
        cid = self.get_pubchem_cid(smiles)
        if cid is not None:
            return self.get_hetcode_from_cid(cid)
        return None

    def _add_res_info(self, m, code, name=None):
        """This code is unnecesarilly big considering i only want to add
        the HET code to the molecule, but there is a bug in RDKIT that screws the
        PDB file if all the other attributes are not added.
        Updating residue Name from UNL to the HET code present in pubchem
        Everythin else (number, is heteroatom, etc) is kept the same but needs to be
        added to avoid the bug.
        See: https://github.com/rdkit/rdkit/pull/7286#issue-2200600916"""
        block = AllChem.MolToPDBBlock(m)
        for line in block.split("\n"):
            if line.startswith("HETATM"):  # avoiding CONECT, COMPND, etc.
                atom_number = line[6:11]
                atom_name = line[12:16]
                res_number = line[22:26]
                atom = m.GetAtomWithIdx(int(atom_number) - 1)
                res_inf = Chem.AtomPDBResidueInfo()
                res_inf.SetName(f"{atom_name}")
                res_inf.SetResidueName(f"{code}")
                res_inf.SetResidueNumber(int(res_number))
                res_inf.SetIsHeteroAtom(True)
                atom.SetPDBResidueInfo(res_inf)
        if name:
            m.SetProp("_Name", name)

    def small_molecule_pdb(self, mol_str: str) -> str:
        # takes in molecule name or smiles (converts to smiles if name)
        # writes pdb file name.pdb (gets name from smiles if possible)
        # output is done message
        ps = Chem.SmilesParserParams()
        ps.removeHs = False
        try:
            if self.is_smiles(mol_str):
                m = Chem.MolFromSmiles(mol_str)
                mol_name = self.smiles2name(mol_str)
                HET_code = self.get_hetcode(mol_str)
                if not HET_code:
                    HET_code = "UNK"
            else:  # if input is not smiles, try getting smiles
                smi = self.molname2smiles(mol_str)
                m = Chem.MolFromSmiles(smi)
                mol_name = mol_str
                HET_code = self.get_hetcode(smi)
                if not HET_code:
                    HET_code = "UNL"

            try:  # only if needed
                m = Chem.AddHs(m)

            except Exception:
                pass

            AllChem.EmbedMolecule(m)
            # add HET code to molecule

            file_name = f"{self.path_registry.ckpt_pdb}/{mol_name}.pdb"
            if HET_code != "UNL":  # if HET code is UNL no need for this...
                self._add_res_info(m, HET_code, mol_name)

            Chem.MolToPDBFile(m, file_name)
            print("finished writing pdb file")
            self.path_registry.map_path(
                mol_name, file_name, f"pdb file for the small molecule {mol_name}"
            )
            return (
                f"Succeeded. PDB file for {mol_str} "
                "successfully created and saved "
                f"to {mol_name}.pdb."
            )
        except Exception as e:
            print(
                "There was an error getting pdb. Please input a single molecule name."
                f"{mol_str},{mol_name}"
            )
            return (
                "Failed. There was an error getting pdb. "
                "Please input a single molecule name."
                "Error: " + str(e)
            )


class SmallMolPDB(BaseTool):
    name = "SmallMoleculePDB"
    description = (
        "Creates a PDB file for a small molecule"
        "Use this tool when you need to use a small molecule in a simulation."
        "Input can be a molecule name or a SMILES string."
    )
    path_registry: Optional[PathRegistry]

    def __init__(self, path_registry: Optional[PathRegistry]):
        super().__init__()
        self.path_registry = path_registry

    def _run(self, mol_str: str) -> str:
        """use the tool."""
        mol_pdb = MolPDB(self.path_registry)
        output = mol_pdb.small_molecule_pdb(mol_str)
        return output
