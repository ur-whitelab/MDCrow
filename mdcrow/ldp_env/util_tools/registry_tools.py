# defining tools


from mdcrow.ldp_env.state import MDCrowState


def MapPath2Name(state: MDCrowState, file: str, path: str):
    """
    Stores the path in the registry with the name provided in the filename.
    If the output says Path mapped to name, then it was successful.
    You do not need to check that file was created.

    Args:
        state (MDCrowState): The state of the MDCrow environment.
        file (str): The desired filename.
        path (str): The file's path.
    """

    path_registry = state.path_registry

    try:
        if path_registry is None:
            return "Failed. Path registry not initialized", 0, False

        map_name = path_registry.map_path(file, path)
        return "Succeeded. " + map_name, 0, False
    except Exception:
        return "Failed. Error writing paths to file", 0, False


def ListRegistryPaths(state, paths: str):
    """Use this tool to list all paths saved in memory.
    Input the word 'paths' and the tool will return a list of all names
    in the registry that are mapped to paths.

    Args:
        state (MDCrowState): The state of the MDCrow environment.
        paths (str): The input string, should be 'paths'.
    """

    path_registry = state.path_registry

    try:
        if path_registry is None:
            return "Failed. Path registry not initialized", 0, False
        return (
            "Succeeded. " + path_registry.list_path_names_and_descriptions(),
            0,
            False,
        )
    except Exception:
        return "Failed. Error listing paths", 0, False
