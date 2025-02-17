import json
import os
import shutil
import tempfile
from datetime import datetime
from unittest.mock import mock_open, patch

import pytest

from mdcrow.agent.agent import MDCrow
from mdcrow.utils import FileType, PathRegistry, SetCheckpoint


@pytest.fixture
def todays_date():
    return str(datetime.today().strftime("%Y%m%d"))


@pytest.fixture()
def set_ckpt():
    return SetCheckpoint()


def clear_ckpt(child: str = "ckpt_test"):
    for dir in os.listdir("ckpt"):
        if child in dir:
            shutil.rmtree(os.path.join("ckpt", dir))


def test_setckpt_root_dir(set_ckpt):
    root_dir = set_ckpt.find_root_dir()
    assert root_dir
    assert "setup.py" in os.listdir(root_dir)


def test_write_to_file(get_registry):
    registry = get_registry("raw", False)
    with patch("builtins.open", mock_open()):
        file_name = registry.write_file_name(
            FileType.PROTEIN,
            protein_name="1XYZ",
            description="testing",
            file_format="pdb",
        )
        # assert file name starts and ends correctly
        assert file_name.startswith("1XYZ")
        assert file_name.endswith(".pdb")


def test_write_file_name_protein(get_registry, todays_date):
    registry = get_registry("raw", False)
    file_name = registry.write_file_name(
        FileType.PROTEIN, protein_name="1XYZ", description="testing", file_format="pdb"
    )
    assert "1XYZ_testing" in file_name
    assert todays_date in file_name
    assert file_name.endswith(".pdb")


def test_write_file_name_simulation_with_conditions(get_registry, todays_date):
    registry = get_registry("raw", False)
    file_name = registry.write_file_name(
        FileType.SIMULATION,
        type_of_sim="MD",
        protein_file_id="1XYZ",
        conditions="pH7",
        time_stamp=todays_date,
    )
    assert "MD_1XYZ_pH7" in file_name
    assert todays_date in file_name
    assert file_name.endswith(".py")


def test_write_file_name_simulation_modified(get_registry, todays_date):
    registry = get_registry("raw", False)
    file_name = registry.write_file_name(
        FileType.SIMULATION, Sim_id="SIM456", modified=True, time_stamp=todays_date
    )
    assert "SIM456_MOD" in file_name
    assert todays_date in file_name
    assert file_name.endswith(".py")


def test_write_file_name_simulation_default(get_registry, todays_date):
    registry = get_registry("raw", False)
    file_name = registry.write_file_name(
        FileType.SIMULATION,
        type_of_sim="MD",
        protein_file_id="123",
        time_stamp=todays_date,
    )
    assert "MD_123" in file_name
    assert todays_date in file_name
    assert file_name.endswith(".py")


def test_write_file_name_record(get_registry, todays_date):
    registry = get_registry("raw", False)
    file_name = registry.write_file_name(
        FileType.RECORD,
        record_type="REC",
        protein_file_id="123",
        Sim_id="SIM456",
        file_format="dcd",
        time_stamp=todays_date,
    )
    assert "REC_SIM456_123" in file_name
    assert todays_date in file_name
    assert file_name.endswith(".dcd")


def test_write_file_name_figure_1(get_registry, todays_date):
    registry = get_registry("raw", False)
    file_name = registry.write_file_name(
        FileType.FIGURE,
        Sim_id="SIM456",
        time_stamp=todays_date,
        file_format="png",
        irrelevant="irrelevant",
    )
    assert "FIG_SIM456_" in file_name
    assert todays_date in file_name
    assert file_name.endswith(".png")


def test_write_file_name_figure_2(get_registry, todays_date):
    registry = get_registry("raw", False)
    file_name = registry.write_file_name(
        FileType.FIGURE,
        Log_id="LOG_123456",
        time_stamp=todays_date,
        file_format="jpg",
        irrelevant="irrelevant",
    )
    assert "FIG_LOG_123456_" in file_name
    assert todays_date in file_name
    assert file_name.endswith(".jpg")


def test_write_file_name_figure_3(get_registry, todays_date):
    registry = get_registry("raw", False)
    file_name = registry.write_file_name(
        FileType.FIGURE,
        Log_id="LOG_123456",
        fig_analysis="randomanalytic",
        file_format="jpg",
        irrelevant="irrelevant",
    )
    assert "FIG_randomanalytic_LOG_123456_" in file_name
    assert todays_date in file_name
    assert file_name.endswith(".jpg")


def test_map_path(get_registry):
    registry = get_registry("raw", False)
    mock_json_data = {
        "existing_name": {
            "path": "existing/path",
            "name": "path",
            "description": "Existing description",
        }
    }
    new_path_dict = {
        "new_name": {
            "path": "new/path",
            "name": "path",
            "description": "New description",
        }
    }
    updated_json_data = {**mock_json_data, **new_path_dict}

    registry.json_file_path = "dummy_json_file.json"

    # Mocking os.path.exists to simulate the JSON file existence
    with patch("os.path.exists", return_value=True):
        # Mocking open for both reading and writing the JSON file
        with patch(
            "builtins.open", mock_open(read_data=json.dumps(mock_json_data))
        ) as mocked_file:
            # Optionally, you can mock internal methods if needed
            with patch.object(
                registry, "_check_for_json", return_value=True
            ), patch.object(
                registry, "_check_json_content", return_value=True
            ), patch.object(
                registry, "_get_full_path", return_value="new/path"
            ):  # Mocking _get_full_path
                result = registry.map_path("new_name", "new/path", "New description")
                # Aggregating all calls to write into a single string
                written_data = "".join(
                    call.args[0] for call in mocked_file().write.call_args_list
                )

                # Comparing the aggregated data with the expected JSON data
                assert json.loads(written_data) == updated_json_data

                # Check the result message
                assert result == "Path successfully mapped to name: new_name"
    if os.path.exists(registry.json_file_path):
        os.remove(registry.json_file_path)


def test_init_path_registry(get_registry):
    registry = get_registry("raw", False)
    temp_file, temp_path = tempfile.mkstemp()
    registry.map_path("temp_path", str(temp_path), "temp file")
    assert "temp_path" in registry.list_path_names()
    os.close(temp_file)


def test_path_registry_ckpt(get_registry):
    registry = get_registry("raw", False)
    ckpt_dir = registry.ckpt_dir
    ckpt_files = registry.ckpt_files
    ckpt_figures = registry.ckpt_figures
    ckpt_pdb = registry.ckpt_pdb
    ckpt_simulations = registry.ckpt_simulations
    ckpt_records = registry.ckpt_records
    all_ckpts = [
        ckpt_dir,
        ckpt_files,
        ckpt_figures,
        ckpt_pdb,
        ckpt_simulations,
        ckpt_records,
    ]

    for ckpt in all_ckpts:
        assert ckpt
        assert os.path.exists(ckpt)
        assert os.path.isdir(ckpt)


def test_mdcrow_w_ckpt():
    dummy_test_dir = "ckpt_test"
    mdcrow = MDCrow(ckpt_dir=dummy_test_dir)
    dummy_test_path = mdcrow.path_registry.ckpt_dir
    assert os.path.exists(dummy_test_path)
    assert dummy_test_dir in dummy_test_path


@pytest.fixture
def root_dir(set_ckpt):
    return set_ckpt.find_root_dir()


def test_setckpt_make_ckpt_parent_folder(set_ckpt, root_dir):
    dir_test = "nonsense_dir"
    # test make_ckpt_parent_folder
    test_ckpt_dir = os.path.join(root_dir, dir_test)
    if os.path.exists(test_ckpt_dir):
        shutil.rmtree(test_ckpt_dir)
    ckpt_path = set_ckpt.make_ckpt_parent_folder(dir_test)
    assert os.path.exists(ckpt_path)
    shutil.rmtree(os.path.join(root_dir, dir_test))


def test_set_ckpt_subdir_single(set_ckpt, root_dir):
    dir_test = "ckpt_test"
    default = "ckpt"

    ckpt_subdir_1 = set_ckpt.set_ckpt_subdir(ckpt_dir=dir_test)
    expected_subdir_1 = os.path.join(root_dir, f"{default}/{dir_test}")
    assert os.path.exists(ckpt_subdir_1)
    assert expected_subdir_1 in ckpt_subdir_1
    shutil.rmtree(os.path.dirname(ckpt_subdir_1))

    ckpt_subdir_2 = set_ckpt.set_ckpt_subdir(ckpt_parent_folder=dir_test)
    expected_subdir_2 = os.path.join(root_dir, f"{dir_test}/{default}")
    assert os.path.exists(ckpt_subdir_2)
    assert expected_subdir_2 in ckpt_subdir_2
    shutil.rmtree(os.path.dirname(ckpt_subdir_2))

    ckpt_subdir_3 = set_ckpt.set_ckpt_subdir()
    expected_subdir_3 = os.path.join(root_dir, f"{default}/{default}")
    assert os.path.exists(ckpt_subdir_3)
    assert expected_subdir_3 in ckpt_subdir_3
    shutil.rmtree(os.path.dirname(ckpt_subdir_3))


def test_set_ckpt_subdir_multiple(set_ckpt, root_dir):
    dir_test = "ckpt_test"
    default = "ckpt"
    expected_subdir_0 = os.path.join(root_dir, f"{dir_test}/{default}_")

    ckpt_subdir_0 = set_ckpt.set_ckpt_subdir(ckpt_parent_folder=dir_test)

    ckpt_subdir_1 = set_ckpt.set_ckpt_subdir(ckpt_parent_folder=dir_test)

    assert int(ckpt_subdir_1.split("_")[-1]) > int(ckpt_subdir_0.split("_")[-1])
    assert expected_subdir_0 in ckpt_subdir_0
    assert expected_subdir_0 in ckpt_subdir_1
    shutil.rmtree(os.path.join(root_dir, dir_test))


def test_path_registry_w_ckpt():
    ckpt_dir = "ckpt_test"
    path_registry = PathRegistry(ckpt_dir=ckpt_dir)
    assert os.path.exists(path_registry.json_file_path)
    assert path_registry.json_file_path.endswith("paths_registry.json")
    assert f"{ckpt_dir}" in os.path.basename(
        os.path.dirname(path_registry.json_file_path)
    )
    shutil.rmtree(os.path.dirname(path_registry.json_file_path))


def test_get_iteration_number():
    pass
