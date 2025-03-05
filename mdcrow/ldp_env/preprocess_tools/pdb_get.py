from typing import Dict, Optional
import os
import requests
from preprocess_tools.path_registry import PathRegistry

def download_pdb_file(
            self,
            query_string: str, 
            path_registry: Optional[PathRegistry] = None
        ) -> Dict[str, Optional[str]]:
        """
        Searches RCSB's Protein Data Bank using the given query string and downloads \
            the corresponding PDB or CIF file.

        Args:
            query_string (str): The search term for querying the PDB database.
            path_registry (Optional[PathRegistry]): An instance of PathRegistry to \
                manage file paths.

        Returns:
            Dict[str, Optional[str]]: A dictionary containing the filename and \
                file ID if successful, otherwise None.
        """
        # if path_registry is None:
        #     path_registry = PathRegistry.get_instance()
        
        url = "https://search.rcsb.org/rcsbsearch/v2/query?json={search-request}"
        query = {
            "query": 
            {
                "type": "terminal", "service": "full_text", "parameters": 
                {
                  "value": query_string
                }
            },
            "return_type": "entry"
        }
        response = requests.post(url, json=query)
        
        if response.status_code == 204:
            return {"filename": None, "file_id": None}
        
        filetype = "cif" if "cif" in query_string.lower() else "pdb"
        
        results = response.json().get("result_set", [])
        if results:
            pdbid = max(results, key=lambda x: x["score"])["identifier"]
            download_url = f"https://files.rcsb.org/download/{pdbid}.{filetype}"
            pdb_response = requests.get(download_url)
            
            filename = path_registry.write_file_name(
                                                "PROTEIN", 
                                                protein_name=pdbid,
                                                description="raw",
                                                file_format=filetype
                                                )
            file_id = path_registry.get_fileid(filename, "PROTEIN")
            directory = path_registry.ckpt_pdb
        #get current directory
            filename = f"{pdbid}.{filetype}"
            directory = "pdb_files"
            file_id = f"{pdbid}.{filetype}"
            # Write the PDB file to disk, if file not already exists
            if not os.path.exists(directory):
                os.makedirs(directory)
            with open(f"{directory}/{filename}", "w") as file:
                file.write(pdb_response.text)
            
            path_registry.map_path(
                file_id,
                f"{directory}/{filename}",
                "PDB file downloaded from RSCB"
                )
            
            return filename, self.config.tool_success_reward, False
        
        return filename, self.config.tool_failure_reward, False