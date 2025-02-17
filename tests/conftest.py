import os
import shutil
from pathlib import Path

import mdtraj as md
import numpy as np
import pytest

from mdcrow.utils import PathRegistry


def safe_remove(file_path):
    """Remove a file if it exists."""
    try:
        os.remove(file_path)
    except FileNotFoundError:
        pass  # the file does not exist, no action needed


@pytest.fixture
def path_to_cif():
    # Save original working directory
    original_cwd = os.getcwd()

    # Change current working directory to the directory where the CIF file is located
    tests_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(tests_dir)

    # Yield the filename only
    filename_only = "3pqr.cif"
    yield filename_only

    # Restore original working directory after the test is done
    os.chdir(original_cwd)


@pytest.fixture
def loaded_cif_traj():
    tests_dir = os.path.dirname(os.path.abspath(__file__))
    cif_path = os.path.join(tests_dir, "3pqr.cif")
    return md.load(cif_path)


@pytest.fixture(scope="module")
def raw_alanine_pdb_file(request):
    pdb_content = """
    ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 20.00           N
    ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00 20.00           C
    ATOM      3  C   ALA A   1       2.175   1.395   0.000  1.00 20.00           C
    ATOM      4  O   ALA A   1       1.461   2.400   0.000  1.00 20.00           O
    ATOM      5  CB  ALA A   1       1.958  -0.735  -1.231  1.00 20.00           C
    TER
    END
        """.strip()
    with open("ALA_raw_123456.pdb", "w") as f:
        f.write(pdb_content)
    yield "ALA_raw_123456.pdb"

    request.addfinalizer(lambda: os.remove("ALA_raw_123456.pdb"))


@pytest.fixture(scope="module")
def clean_alanine_pdb_file(request):
    pdb_content = """
REMARK  ACE
CRYST1   32.155   32.155   56.863  90.00  90.00 120.00 P 31 2 1      6
ATOM      1 1HH3 ACE     1       2.000   1.000  -0.000
ATOM      2  CH3 ACE     1       2.000   2.090   0.000
ATOM      3 2HH3 ACE     1       1.486   2.454   0.890
ATOM      4 3HH3 ACE     1       1.486   2.454  -0.890
ATOM      5  C   ACE     1       3.427   2.641  -0.000
ATOM      6  O   ACE     1       4.391   1.877  -0.000
ATOM      7  N   ALA     2       3.555   3.970  -0.000
ATOM      8  H   ALA     2       2.733   4.556  -0.000
ATOM      9  CA  ALA     2       4.853   4.614  -0.000
ATOM     10  HA  ALA     2       5.408   4.316   0.890
ATOM     11  CB  ALA     2       5.661   4.221  -1.232
ATOM     12 1HB  ALA     2       5.123   4.521  -2.131
ATOM     13 2HB  ALA     2       6.630   4.719  -1.206
ATOM     14 3HB  ALA     2       5.809   3.141  -1.241
ATOM     15  C   ALA     2       4.713   6.129   0.000
ATOM     16  O   ALA     2       3.601   6.653   0.000
ATOM     17  N   NME     3       5.846   6.835   0.000
ATOM     18  H   NME     3       6.737   6.359  -0.000
ATOM     19  CH3 NME     3       5.846   8.284   0.000
ATOM     20 1HH3 NME     3       4.819   8.648   0.000
ATOM     21 2HH3 NME     3       6.360   8.648   0.890
ATOM     22 3HH3 NME     3       6.360   8.648  -0.890
TER
END
        """
    with open("ALA_clean_654321.pdb", "w") as f:
        f.write(pdb_content)

    yield "ALA_clean_654321.pdb"

    request.addfinalizer(lambda: os.remove("ALA_clean_654321.pdb"))


@pytest.fixture(scope="module")
def ALA_CIF_file(request):
    cif_content = """
    data_ALA
#
_chem_comp.id                                    ALA
_chem_comp.name                                  ALANINE
_chem_comp.type                                  "L-PEPTIDE LINKING"
_chem_comp.pdbx_type                             ATOMP
_chem_comp.formula                               "C3 H7 N O2"
_chem_comp.mon_nstd_parent_comp_id               ?
_chem_comp.pdbx_synonyms                         ?
_chem_comp.pdbx_formal_charge                    0
_chem_comp.pdbx_initial_date                     1999-07-08
_chem_comp.pdbx_modified_date                    2023-11-03
_chem_comp.pdbx_ambiguous_flag                   N
_chem_comp.pdbx_release_status                   REL
_chem_comp.pdbx_replaced_by                      ?
_chem_comp.pdbx_replaces                         ?
_chem_comp.formula_weight                        89.093
_chem_comp.one_letter_code                       A
_chem_comp.three_letter_code                     ALA
_chem_comp.pdbx_model_coordinates_details        ?
_chem_comp.pdbx_model_coordinates_missing_flag   N
_chem_comp.pdbx_ideal_coordinates_details        ?
_chem_comp.pdbx_ideal_coordinates_missing_flag   N
_chem_comp.pdbx_model_coordinates_db_code        ?
_chem_comp.pdbx_subcomponent_list                ?
_chem_comp.pdbx_processing_site                  RCSB
#
loop_
_chem_comp_atom.comp_id
_chem_comp_atom.atom_id
_chem_comp_atom.alt_atom_id
_chem_comp_atom.type_symbol
_chem_comp_atom.charge
_chem_comp_atom.pdbx_align
_chem_comp_atom.pdbx_aromatic_flag
_chem_comp_atom.pdbx_leaving_atom_flag
_chem_comp_atom.pdbx_stereo_config
_chem_comp_atom.pdbx_backbone_atom_flag
_chem_comp_atom.pdbx_n_terminal_atom_flag
_chem_comp_atom.pdbx_c_terminal_atom_flag
_chem_comp_atom.model_Cartn_x
_chem_comp_atom.model_Cartn_y
_chem_comp_atom.model_Cartn_z
_chem_comp_atom.pdbx_model_Cartn_x_ideal
_chem_comp_atom.pdbx_model_Cartn_y_ideal
_chem_comp_atom.pdbx_model_Cartn_z_ideal
_chem_comp_atom.pdbx_component_atom_id
_chem_comp_atom.pdbx_component_comp_id
_chem_comp_atom.pdbx_ordinal
ALA N   N   N 0 1 N N N Y Y N 2.281  26.213 12.804 -0.966 0.493  1.500  N   ALA 1
ALA CA  CA  C 0 1 N N S Y N N 1.169  26.942 13.411 0.257  0.418  0.692  CA  ALA 2
ALA C   C   C 0 1 N N N Y N Y 1.539  28.344 13.874 -0.094 0.017  -0.716 C   ALA 3
ALA O   O   O 0 1 N N N Y N Y 2.709  28.647 14.114 -1.056 -0.682 -0.923 O   ALA 4
ALA CB  CB  C 0 1 N N N N N N 0.601  26.143 14.574 1.204  -0.620 1.296  CB  ALA 5
ALA OXT OXT O 0 1 N Y N Y N Y 0.523  29.194 13.997 0.661  0.439  -1.742 OXT ALA 6
ALA H   H   H 0 1 N N N Y Y N 2.033  25.273 12.493 -1.383 -0.425 1.482  H   ALA 7
ALA H2  HN2 H 0 1 N Y N Y Y N 3.080  26.184 13.436 -0.676 0.661  2.452  H2  ALA 8
ALA HA  HA  H 0 1 N N N Y N N 0.399  27.067 12.613 0.746  1.392  0.682  HA  ALA 9
ALA HB1 1HB H 0 1 N N N N N N -0.247 26.699 15.037 1.459  -0.330 2.316  HB1 ALA 10
ALA HB2 2HB H 0 1 N N N N N N 0.308  25.110 14.270 0.715  -1.594 1.307  HB2 ALA 11
ALA HB3 3HB H 0 1 N N N N N N 1.384  25.876 15.321 2.113  -0.676 0.697  HB3 ALA 12
ALA HXT HXT H 0 1 N Y N Y N Y 0.753  30.069 14.286 0.435  0.182  -2.647 HXT ALA 13
#
loop_
_chem_comp_bond.comp_id
_chem_comp_bond.atom_id_1
_chem_comp_bond.atom_id_2
_chem_comp_bond.value_order
_chem_comp_bond.pdbx_aromatic_flag
_chem_comp_bond.pdbx_stereo_config
_chem_comp_bond.pdbx_ordinal
ALA N   CA  SING N N 1
ALA N   H   SING N N 2
ALA N   H2  SING N N 3
ALA CA  C   SING N N 4
ALA CA  CB  SING N N 5
ALA CA  HA  SING N N 6
ALA C   O   DOUB N N 7
ALA C   OXT SING N N 8
ALA CB  HB1 SING N N 9
ALA CB  HB2 SING N N 10
ALA CB  HB3 SING N N 11
ALA OXT HXT SING N N 12
#
loop_
_pdbx_chem_comp_descriptor.comp_id
_pdbx_chem_comp_descriptor.type
_pdbx_chem_comp_descriptor.program
_pdbx_chem_comp_descriptor.program_version
_pdbx_chem_comp_descriptor.descriptor
ALA SMILES           ACDLabs              10.04 "O=C(O)C(N)C"
ALA SMILES_CANONICAL CACTVS               3.341 "C[C@H](N)C(O)=O"
ALA SMILES           CACTVS               3.341 "C[CH](N)C(O)=O"
ALA SMILES_CANONICAL "OpenEye OEToolkits" 1.5.0 "C[C@@H](C(=O)O)N"
ALA SMILES           "OpenEye OEToolkits" 1.5.0 "CC(C(=O)O)N"
ALA InChI            InChI                1.03  "InChI=1S/C3H7NO2/c1-2(4)3(5)6/h2H,4H2,\
1H3,(H,5,6)/t2-/m0/s1"
ALA InChIKey         InChI                1.03  QNAYBMKLOCPYGJ-REOHCLBHSA-N
#
loop_
_pdbx_chem_comp_identifier.comp_id
_pdbx_chem_comp_identifier.type
_pdbx_chem_comp_identifier.program
_pdbx_chem_comp_identifier.program_version
_pdbx_chem_comp_identifier.identifier
ALA "SYSTEMATIC NAME" ACDLabs              10.04 L-alanine
ALA "SYSTEMATIC NAME" "OpenEye OEToolkits" 1.5.0 "(2S)-2-aminopropanoic acid"
#
loop_
_pdbx_chem_comp_audit.comp_id
_pdbx_chem_comp_audit.action_type
_pdbx_chem_comp_audit.date
_pdbx_chem_comp_audit.processing_site
ALA "Create component"  1999-07-08 RCSB
ALA "Modify descriptor" 2011-06-04 RCSB
ALA "Modify backbone"   2023-11-03 PDBE
#
"""
    with open("ALA_clean_654321.cif", "w") as f:
        f.write(cif_content)
    yield "ALA_clean_654321.cif"

    request.addfinalizer(lambda: os.remove("ALA_clean_654321.cif"))


def setup_butane_topology(include_hydrogens=False):
    n_carbons = 4
    topology = md.Topology()
    chain = topology.add_chain()
    residue = topology.add_residue("BUT", chain)

    carbon_atoms = [
        topology.add_atom("C", md.element.carbon, residue) for _ in range(n_carbons)
    ]
    for i in range(n_carbons - 1):
        topology.add_bond(carbon_atoms[i], carbon_atoms[i + 1])

    if not include_hydrogens:
        return topology, carbon_atoms, None

    hydrogens_per_carbon = [3, 2, 2, 3]
    hydrogen_atoms = []
    for i, carbon in enumerate(carbon_atoms):
        for _ in range(hydrogens_per_carbon[i]):
            h_atom = topology.add_atom("H", md.element.hydrogen, residue)
            hydrogen_atoms.append(h_atom)
            topology.add_bond(carbon, h_atom)
    return topology, carbon_atoms, hydrogen_atoms


@pytest.fixture(scope="module")
def butane_static_trajectory_with_hydrogens(request):
    topology, carbon_atoms, hydrogen_atoms = setup_butane_topology(
        include_hydrogens=True
    )
    n_frames = 100
    xyz = np.zeros((n_frames, len(carbon_atoms) + len(hydrogen_atoms), 3))

    # set carbon positions along the x-axis, 1.54 Å apart
    carbon_positions = np.array([[i * 1.54, 0, 0] for i in range(len(carbon_atoms))])
    # define hydrogen positions around each carbon
    hydrogen_offsets = [
        [-0.77, 0.77, 0],
        [0, -0.77, 0.77],
        [0.77, 0.77, 0],  # H for C1
        [-0.77, 0, 0.77],
        [0.77, 0, 0.77],  # H for C2
        [-0.77, 0, -0.77],
        [0.77, 0, -0.77],  # H for C3
        [-0.77, -0.77, 0],
        [0, -0.77, -0.77],
        [0.77, -0.77, 0],  # H for C4
    ]
    for i in range(n_frames):
        xyz[i, : len(carbon_atoms), :] = carbon_positions
        for j, h_offset in enumerate(hydrogen_offsets):
            xyz[i, len(carbon_atoms) + j, :] = carbon_positions[j // 3] + h_offset

    trajectory = md.Trajectory(xyz=xyz, topology=topology)
    dcd_file = "TRAJ_butane_with_hydrogens_123456.dcd"
    pdb_file = "TOP_butane_with_hydrogens_123456.pdb"
    trajectory.save_dcd(dcd_file)
    trajectory[0].save_pdb(pdb_file)  # save the first frame as a PDB for topology

    yield dcd_file, pdb_file
    request.addfinalizer(lambda: safe_remove(dcd_file))
    request.addfinalizer(lambda: safe_remove(pdb_file))


@pytest.fixture(scope="module")
def butane_dynamic_trajectory_with_hydrogens(request):
    topology, carbon_atoms, hydrogen_atoms = setup_butane_topology(
        include_hydrogens=True
    )
    n_frames = 100
    xyz = np.zeros((n_frames, len(carbon_atoms) + len(hydrogen_atoms), 3))

    # set carbon positions along the x-axis, 1.54 Å apart
    carbon_positions = np.array([[i * 1.54, 0, 0] for i in range(len(carbon_atoms))])
    # define hydrogen positions around each carbon
    hydrogen_offsets = [
        [-0.77, 0.77, 0],
        [0, -0.77, 0.77],
        [0.77, 0.77, 0],  # H for C1
        [-0.77, 0, 0.77],
        [0.77, 0, 0.77],  # H for C2
        [-0.77, 0, -0.77],
        [0.77, 0, -0.77],  # H for C3
        [-0.77, -0.77, 0],
        [0, -0.77, -0.77],
        [0.77, -0.77, 0],  # H for C4
    ]

    for i in range(n_frames):
        xyz[i, : len(carbon_atoms), :] = carbon_positions + np.random.normal(
            scale=0.1, size=carbon_positions.shape
        )
        for j, h_offset in enumerate(hydrogen_offsets):
            xyz[i, len(carbon_atoms) + j, :] = (
                carbon_positions[j // 3]
                + h_offset
                + np.random.normal(scale=0.1, size=(3,))
            )

    trajectory = md.Trajectory(xyz=xyz, topology=topology)
    dcd_file = "TRAJ_butane_with_hydrogens_123456.dcd"
    pdb_file = "TOP_butane_with_hydrogens_123456.pdb"
    trajectory.save_dcd(dcd_file)
    trajectory[0].save_pdb(pdb_file)  # save the first frame as a PDB for topology

    yield dcd_file, pdb_file
    request.addfinalizer(lambda: safe_remove(dcd_file))
    request.addfinalizer(lambda: safe_remove(pdb_file))


@pytest.fixture(scope="module")
def butane_trajectory_without_hydrogens(request):
    topology, carbon_atoms, _ = setup_butane_topology(include_hydrogens=False)
    n_frames = 100
    xyz = np.zeros((n_frames, len(carbon_atoms), 3))

    # set carbon positions along the x-axis, 1.54 Å apart
    carbon_positions = np.array([[i * 1.54, 0, 0] for i in range(len(carbon_atoms))])
    for i in range(n_frames):
        xyz[i, :, :] = carbon_positions

    trajectory = md.Trajectory(xyz=xyz, topology=topology)
    dcd_file = "TRAJ_butane_without_hydrogens_123456.dcd"
    pdb_file = "TOP_butane_without_hydrogens_123456.pdb"
    trajectory.save_dcd(dcd_file)
    trajectory[0].save_pdb(pdb_file)  # save the first frame as a PDB for topology

    yield dcd_file, pdb_file
    request.addfinalizer(lambda: safe_remove(dcd_file))
    request.addfinalizer(lambda: safe_remove(pdb_file))


@pytest.fixture(scope="module")
def small_peptide_gag_trajectory(request):
    """
    Writes out a small multi-model PDB with two frames of a Gly–Ala–Gly peptide.
    Yields the filename so that tests can load and analyze phi/psi angles.
    """
    pdb_content = """\
MODEL        1
ATOM      1  N   GLY A   1       1.000   1.000   1.000  1.00  0.00           N
ATOM      2  CA  GLY A   1       2.000   1.000   1.000  1.00  0.00           C
ATOM      3  C   GLY A   1       2.500   2.250   1.000  1.00  0.00           C
ATOM      4  O   GLY A   1       3.000   3.000   1.000  1.00  0.00           O
ATOM      5  N   ALA A   2       1.700   2.900   1.000  1.00  0.00           N
ATOM      6  CA  ALA A   2       2.000   4.300   1.100  1.00  0.00           C
ATOM      7  CB  ALA A   2       1.000   5.250   1.000  1.00  0.00           C
ATOM      8  C   ALA A   2       3.400   5.000   1.000  1.00  0.00           C
ATOM      9  O   ALA A   2       4.100   5.900   1.500  1.00  0.00           O
ATOM     10  N   GLY A   3       3.900   4.300   0.200  1.00  0.00           N
ATOM     11  CA  GLY A   3       5.200   4.850   0.000  1.00  0.00           C
ATOM     12  C   GLY A   3       6.100   3.700   0.000  1.00  0.00           C
ATOM     13  O   GLY A   3       7.000   3.700   0.700  1.00  0.00           O
ENDMDL
MODEL        2
ATOM      1  N   GLY A   1       1.100   1.000   0.900  1.00  0.00           N
ATOM      2  CA  GLY A   1       2.050   1.100   1.100  1.00  0.00           C
ATOM      3  C   GLY A   1       2.550   2.300   1.100  1.00  0.00           C
ATOM      4  O   GLY A   1       3.050   3.100   1.050  1.00  0.00           O
ATOM      5  N   ALA A   2       1.650   2.900   1.100  1.00  0.00           N
ATOM      6  CA  ALA A   2       2.000   4.300   1.100  1.00  0.00           C
ATOM      7  CB  ALA A   2       1.200   5.350   1.050  1.00  0.00           C
ATOM      8  C   ALA A   2       3.450   4.900   1.050  1.00  0.00           C
ATOM      9  O   ALA A   2       4.150   5.800   1.600  1.00  0.00           O
ATOM     10  N   GLY A   3       3.900   4.200   0.350  1.00  0.00           N
ATOM     11  CA  GLY A   3       5.300   4.850   0.300  1.00  0.00           C
ATOM     12  C   GLY A   3       6.200   3.650   0.250  1.00  0.00           C
ATOM     13  O   GLY A   3       7.100   3.600   0.900  1.00  0.00           O
ENDMDL
END
"""

    # You can name this file however you want; using a random suffix is common.
    pdb_filename = "SMALL_GAG_trajectory_987654.pdb"
    with open(pdb_filename, "w") as f:
        f.write(pdb_content)

    # Yield the filename to the test function(s).
    yield pdb_filename

    # Cleanup after tests complete
    request.addfinalizer(lambda: safe_remove(pdb_filename))


@pytest.fixture(scope="module")
def get_registry(
    raw_alanine_pdb_file,
    clean_alanine_pdb_file,
    butane_static_trajectory_with_hydrogens,
    butane_dynamic_trajectory_with_hydrogens,
    butane_trajectory_without_hydrogens,
    small_peptide_gag_trajectory,
    request,
):
    created_paths = []  # Keep track of created directories for cleanup

    def get_new_ckpt():
        registry = PathRegistry.get_instance(ckpt_dir="ckpt_test")
        base_path = registry.ckpt_dir
        return base_path, registry

    def create(
        raw_or_clean,
        with_files,
        dynamic=False,
        include_hydrogens=False,
        map_path=True,
        include_peptide_trajectory=False,
    ):
        base_path, registry = get_new_ckpt()
        created_paths.append(base_path)
        if with_files:
            pdb_path = Path(base_path) / "pdb"
            record_path = Path(base_path) / "records"
            simulation_path = Path(base_path) / "simulation"
            figure_path = Path(base_path) / "figures"

            # Create directories
            for path in [pdb_path, record_path, simulation_path, figure_path]:
                os.makedirs(path, exist_ok=True)
                created_paths.append(path)
            # create files
            files = {}
            if raw_or_clean == "raw":
                # Copy the alanine pdb file to the pdb/alanine directory
                files["ALA_123456"] = {"name": raw_alanine_pdb_file, "dir": pdb_path}
            elif raw_or_clean == "clean":
                files["ALA_654321"] = {"name": clean_alanine_pdb_file, "dir": pdb_path}
            if include_hydrogens:
                if dynamic:
                    traj_file, top_file = butane_dynamic_trajectory_with_hydrogens
                else:
                    traj_file, top_file = butane_static_trajectory_with_hydrogens
            else:
                traj_file, top_file = butane_trajectory_without_hydrogens
            if include_peptide_trajectory:
                pep_traj_file = small_peptide_gag_trajectory
                files["pep_traj_987654"] = {"name": pep_traj_file, "dir": record_path}

            files["rec0_butane_123456"] = {"name": traj_file, "dir": record_path}
            files["top_sim0_butane_123456"] = {"name": top_file, "dir": pdb_path}
            for f in files:
                shutil.copy(files[f]["name"], files[f]["dir"])
                if map_path:
                    full_path = os.path.join(files[f]["dir"], files[f]["name"])
                    registry.map_path(f, full_path)
        return registry

    # Cleanup: Remove created directories and the copied pdb file
    def cleanup():
        for path in reversed(created_paths):  # Remove directories
            shutil.rmtree(path, ignore_errors=True)
        if os.path.exists("paths_registry.json"):
            os.remove("paths_registry.json")

    request.addfinalizer(cleanup)

    return create
