# defining tools
from typing import Optional

from langchain.tools import BaseTool

from mdcrow.utils import PathRegistry


class MapPath2Name(BaseTool):
    name = "MapPath2Name"
    description = """Input the desired filename
    followed by the file's path, separated by a comma.
    Make sure the name is first, then the path.
    Your path should look something like: name.pdb
    Stores the path in the registry with the
    name provided in the filename.
    If the output says Path mapped to name,
    then it was successful.
    You do not need to check that file was created."""
    path_registry: Optional[PathRegistry]

    def __init__(self, path_registry: Optional[PathRegistry]):
        super().__init__()
        self.path_registry = path_registry

    def _run(self, file_and_path: str) -> str:
        """Use the tool"""
        try:
            if self.path_registry is None:
                return "Failed. Path registry not initialized"
            if "," not in file_and_path:
                return "Failed. Please separate filename and path with a comma"
            file, path = file_and_path.split(",")
            map_name = self.path_registry.map_path(file, path)
            return "Succeeded. " + map_name
        except Exception:
            return "Failed. Error writing paths to file"

    async def _arun(self, file_name: str) -> str:
        """Use the tool asynchronously"""
        raise NotImplementedError


class ListRegistryPaths(BaseTool):
    name = "ListRegistryPaths"
    description = """Use this tool to list all paths saved in memory.
    Input the word 'paths' and the tool will return a list of all names
    in the registry that are mapped to paths."""
    path_registry: Optional[PathRegistry]

    def __init__(self, path_registry: Optional[PathRegistry]):
        super().__init__()
        self.path_registry = path_registry

    def _run(self, paths: str) -> str:
        """Use the tool"""
        try:
            if self.path_registry is None:
                return "Failed. Path registry not initialized"
            return "Succeeded. " + self.path_registry.list_path_names_and_descriptions()
        except Exception:
            return "Failed. Error listing paths"

    async def _arun(self, paths: str) -> str:
        """Use the tool asynchronously"""
        raise NotImplementedError
