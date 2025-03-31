from mdcrow.ldp_env.preprocess_tools.clean_tools import clean_pdb_file
from mdcrow.ldp_env.state import MDCrowState


def test_cleaning_function(get_registry):

    reg = get_registry("raw", True)
    state = MDCrowState(path_registry=reg)
    tool = CleaningToolFunction(path_registry=reg)
    assert state.path_registry
    assert state.path_registry == reg
    prompt = {
        "pdb_id": "ALA_123456",
        "replace_nonstandard_residues": True,
        "add_missing_atoms": True,
        "remove_heterogens": True,
        "remove_water": True,
        "add_hydrogens": True,
        "add_hydrogens_ph": 7.0,
    }
    result, _, _ = clean_pdb_file(state, **prompt)
    assert "File cleaned" in result
