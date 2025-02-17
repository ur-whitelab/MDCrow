from mdcrow.tools.base_tools import CleaningToolFunction


def test_cleaning_function(get_registry):
    reg = get_registry("raw", True)
    tool = CleaningToolFunction(path_registry=reg)
    assert tool.path_registry
    assert tool.name == "CleaningToolFunction"
    assert tool.path_registry == reg
    prompt = {
        "pdb_id": "ALA_123456",
        "replace_nonstandard_residues": True,
        "add_missing_atoms": True,
        "remove_heterogens": True,
        "remove_water": True,
        "add_hydrogens": True,
        "add_hydrogens_ph": 7.0,
    }
    result = tool._run(**prompt)
    assert "File cleaned" in result
