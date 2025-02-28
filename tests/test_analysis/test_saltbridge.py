import mdtraj as md
import pytest

from mdagent.tools.base_tools.analysis_tools.salt_bridge_tool import SaltBridgeFunction

# pdb with salt bridge residues (ARG, ASP, LYS, GLU)
pdb_data = """
HEADER    MOCK SALT BRIDGE EXAMPLE
ATOM      1  N   ARG A   1       0.000   0.000   0.000
ATOM      2  CA  ARG A   1       1.000   0.000   0.000
ATOM      3  C   ARG A   1       1.500   1.000   0.000
ATOM      4  NH1 ARG A   1       2.000   1.500   0.000
ATOM      5  NH2 ARG A   1       2.000   1.800   0.000
ATOM      6  N   LYS A   3       0.000   1.000   1.000
ATOM      7  CA  LYS A   3       1.000   1.000   1.000
ATOM      8  C   LYS A   3       1.500   2.000   1.000
ATOM      9  NZ  LYS A   3       2.000   2.500   1.000
ATOM     10  N   ASP A   2       3.000   1.000   0.000
ATOM     11  CA  ASP A   2       3.500   1.500   0.000
ATOM     12  C   ASP A   2       4.000   2.000   0.000
ATOM     13  OD1 ASP A   2       4.500   2.500   0.000
ATOM     14  OD2 ASP A   2       4.200   2.800   0.000
ATOM     15  N   GLU A   4       2.000   2.000   0.000
ATOM     16  CA  GLU A   4       2.500   2.500   0.000
ATOM     17  C   GLU A   4       3.000   3.000   0.000
ATOM     18  OE1 GLU A   4       3.500   3.500   0.000
ATOM     19  OE2 GLU A   4       3.800   3.800   0.000
END
"""


@pytest.fixture
def get_salt_bridge_function(get_registry):
    # Create the SaltBridgeFunction object using the PDB file path
    reg = get_registry("raw", True)
    pdb_path = f"{reg.ckpt_dir}/sb_residues.pdb"
    with open(pdb_path, "w") as file:
        file.write(pdb_data)
    fxn = SaltBridgeFunction(reg)
    fxn.traj = md.load(pdb_path)
    fxn.traj_file = "sb_residues"
    return fxn


@pytest.fixture
def get_salt_bridge_function_with_butane(get_registry):
    registry = get_registry("raw", True)
    traj_fileid = "rec0_butane_123456"
    top_fileid = "top_sim0_butane_123456"
    fxn = SaltBridgeFunction(registry)
    fxn._load_traj(traj_fileid, top_fileid)
    return fxn


def test_find_salt_bridges_with_salt_bridges(get_salt_bridge_function):
    salt_bridge_function = get_salt_bridge_function
    salt_bridge_function.find_salt_bridges()
    assert len(salt_bridge_function.salt_bridge_counts) == 1
    assert len(salt_bridge_function.salt_bridge_data) == 12
    assert salt_bridge_function.salt_bridge_counts == [12]


def test_salt_bridge_files_single_frame(get_salt_bridge_function):
    salt_bridge_function = get_salt_bridge_function
    salt_bridge_function.find_salt_bridges()
    file_id = salt_bridge_function.save_results_to_file()
    fig_id = salt_bridge_function.plot_salt_bridge_counts()
    assert file_id is not None
    assert fig_id is None


def test_salt_bridge_files_multiple_frames(get_salt_bridge_function):
    salt_bridge_function = get_salt_bridge_function
    n_frames = 5
    multi_frame_traj = md.join([salt_bridge_function.traj] * n_frames)
    salt_bridge_function.traj = multi_frame_traj
    salt_bridge_function.find_salt_bridges()
    file_id = salt_bridge_function.save_results_to_file()
    fig_id = salt_bridge_function.plot_salt_bridge_counts()
    assert file_id is not None
    assert fig_id is not None


def test_no_salt_bridges(get_salt_bridge_function_with_butane):
    salt_bridge_function = get_salt_bridge_function_with_butane
    salt_bridge_function.find_salt_bridges()
    file_id = salt_bridge_function.save_results_to_file()
    fig_id = salt_bridge_function.plot_salt_bridge_counts()
    assert file_id is None
    assert fig_id is None
    assert len(salt_bridge_function.salt_bridge_counts) == 0
    assert len(salt_bridge_function.salt_bridge_data) == 0
    assert salt_bridge_function.salt_bridge_data == []
    assert file_id is None
    assert fig_id is None


def test_invalid_trajectory(get_salt_bridge_function):
    salt_bridge_function = get_salt_bridge_function
    salt_bridge_function.traj = None
    with pytest.raises(Exception, match="MDTrajectory hasn't been loaded"):
        salt_bridge_function.find_salt_bridges()
