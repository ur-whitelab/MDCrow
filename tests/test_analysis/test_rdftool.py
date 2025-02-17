import json

import pytest

from mdcrow.tools.base_tools.analysis_tools.rdf_tool import RDFTool

# TODO add dcd files in testing file for testing


@pytest.fixture(scope="module")
def rdf_input_good_string():
    return """
    {
        "trajectory_fileid": "rec0_142404",
        "topology_fileid": "top_sim0_142401",
        "stride": 2,
        "selections": ["protein", "water"]
    }
    """


@pytest.fixture(scope="module")
def rdf_input_wrong_string_1():
    return """
    {
        "topology_fileid": "top_sim0_142401",
        "stride": 2,
        "selections": [["protein", "water", "lipid"]]
    }
    """


@pytest.fixture(scope="module")
def rdf_input_wrong_string_2():
    return """
    {
        "trajectory_fileid": "rec0_142404",
        "topology_fileid": "top_sim0_142401",
        "stride": "half",
        "selections": [["protein", "water", "lipid"]]
    }
    """


@pytest.fixture(scope="module")
def rdf_input_wrong_string_3():
    return """
    {
        "trajectory_fileid": "rec0_142404Wrong",
        "topology_fileid": "top_sim0_142401",
        "stride": 2,
        "selections": [["protein", "water"]]
    }
    """


def test_rdf_tool_init(get_registry):
    registry = get_registry("raw", False)
    tool = RDFTool(path_registry=registry)
    assert tool.name == "RDFTool"
    assert tool.path_registry == registry


class MockRegistryResponse:
    # mock json() method always returns a specific testing dictionary
    dictionary = {
        "rec0_142404": "files/records/TRAJ_sim0123456_142404.dcd",
        "top_sim0_142401": "files/records/TOP_sim0123456_142401.pdb",
    }

    @staticmethod
    def get_mapped_path(self, fileid):
        return self.dictionary[fileid]

    def list_path_names(self):
        return "rec0_142404", "top_sim0_142401"


def test_rdf_tool_validation(
    monkeypatch,
    rdf_input_good_string,
    rdf_input_wrong_string_1,
    rdf_input_wrong_string_2,
    rdf_input_wrong_string_3,
    get_registry,
):
    registry = get_registry("raw", False)
    tool = RDFTool(path_registry=registry)

    def mock_get_mapped_path(fileid):
        return MockRegistryResponse.dictionary[fileid]

    def mock_list_path_names():
        return "rec0_142404", "top_sim0_142401"

    monkeypatch.setattr(tool.path_registry, "get_mapped_path", mock_get_mapped_path)
    monkeypatch.setattr(tool.path_registry, "list_path_names", mock_list_path_names)
    # assert that a valueerror was raised
    with pytest.raises(ValueError) as error:
        inputs = tool.validate_input(json.loads(rdf_input_wrong_string_1))
        assert str(error.value) == "Missing Inputs: Trajectory file ID is required"
        inputs = tool.validate_input(json.loads(rdf_input_wrong_string_2))
        assert str(error.value) == (
            "Incorrect Inputs: Stride must be an integer "
            "or None for default value of 1"
        )
        inputs = tool.validate_input(json.loads(rdf_input_wrong_string_3))
        assert str(error.value) == "Trajectory File ID not in path registry"

    inputs = tool.validate_input(json.loads(rdf_input_good_string))

    assert inputs["trajectory_fileid"] == "rec0_142404"
    assert inputs["topology_fileid"] == "top_sim0_142401"
    assert inputs["stride"] == 2
    # assert inputs["selections"] == ["protein", "water"]
