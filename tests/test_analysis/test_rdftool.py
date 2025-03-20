import asyncio
import json

import pytest

from mdcrow.ldp_env.analysis_tools.rdf_tools import compute_rdf, validate_input
from mdcrow.ldp_env.state import MDCrowState


@pytest.fixture(scope="module")
def state(get_registry):
    registry = get_registry("raw", False)
    return MDCrowState(path_registry=registry, tools=[])


@pytest.fixture(scope="module")
def rdf_input_good_string():
    return """
    {
        "trajectory_fileid": "rec0_142404",
        "topology_fileid": "top_sim0_142401",
        "stride": 2
    }
    """


@pytest.fixture(scope="module")
def rdf_input_wrong_string_1():
    return """
    {
        "topology_fileid": "top_sim0_142401",
        "stride": 2,
    }
    """


@pytest.fixture(scope="module")
def rdf_input_wrong_string_2():
    return """
    {
        "trajectory_fileid": "rec0_142404",
        "topology_fileid": "top_sim0_142401",
        "stride": "half',
    }
    """


@pytest.fixture(scope="module")
def rdf_input_wrong_string_3():
    return """
    {
        "trajectory_fileid": "rec0_142404Wrong",
        "topology_fileid": "top_sim0_142401",
        "stride": 2,
    }
    """


class MockRegistryResponse:
    # Mock json() method always returns a specific testing dictionary
    dictionary = {
        "rec0_142404": "files/records/TRAJ_sim0123456_142404.dcd",
        "top_sim0_142401": "files/records/TOP_sim0123456_142401.pdb",
    }

    @staticmethod
    def get_mapped_path(fileid):
        return MockRegistryResponse.dictionary.get(fileid, None)

    @staticmethod
    def list_path_names():
        return ["rec0_142404", "top_sim0_142401"]


def test_rdf_tool_validation(
    monkeypatch,
    state,
    rdf_input_good_string,
    rdf_input_wrong_string_1,
    rdf_input_wrong_string_2,
    rdf_input_wrong_string_3,
):
    def mock_get_mapped_path(fileid):
        return MockRegistryResponse.get_mapped_path(fileid)

    def mock_list_path_names():
        return MockRegistryResponse.list_path_names()

    monkeypatch.setattr(state.path_registry, "get_mapped_path", mock_get_mapped_path)
    monkeypatch.setattr(state.path_registry, "list_path_names", mock_list_path_names)

    # Assert that a ValueError was raised
    with pytest.raises(ValueError) as error:
        _ = asyncio.run(compute_rdf(state, **json.loads(rdf_input_wrong_string_1)))
        assert (
            str(error.value)
            == "Incorrect Inputs: Trajectory file ID ('trajectory_fileid') is required"
        )

    with pytest.raises(ValueError) as error:
        _ = asyncio.run(compute_rdf(state, **json.loads(rdf_input_wrong_string_2)))
        assert str(error.value) == (
            "Incorrect Inputs: Stride must be an integer "
            "or None for default value of 1"
        )

    with pytest.raises(ValueError) as error:
        _ = asyncio.run(compute_rdf(state, **json.loads(rdf_input_wrong_string_3)))
        assert str(error.value) == "Trajectory File ID not in path registry"

    print(json.loads(rdf_input_good_string))
    inputs = validate_input(state.path_registry, json.loads(rdf_input_good_string))

    assert inputs["trajectory_fileid"] == "rec0_142404"
    assert inputs["topology_fileid"] == "top_sim0_142401"
    assert inputs["stride"] == 2
