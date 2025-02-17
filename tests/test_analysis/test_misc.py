import os
from unittest.mock import MagicMock, mock_open, patch

import pytest

from mdcrow.tools.base_tools import VisFunctions
from mdcrow.tools.base_tools.analysis_tools.plot_tools import PlottingTools


@pytest.fixture
def plotting_tools(get_registry):
    return PlottingTools(get_registry("raw", False))


@pytest.fixture
def vis_fxns(get_registry):
    return VisFunctions(get_registry("raw", False))


def test_process_csv(plotting_tools):
    mock_csv_content = "Time,Value1,Value2\n1,10,20\n2,15,25"
    mock_reader = MagicMock()
    mock_reader.fieldnames = ["Time", "Value1", "Value2"]
    mock_reader.__iter__.return_value = iter(
        [
            {"Time": "1", "Value1": "10", "Value2": "20"},
            {"Time": "2", "Value1": "15", "Value2": "25"},
        ]
    )
    plotting_tools.file_path = "mock_file.csv"
    plotting_tools.file_name = "mock_file.csv"
    with patch("builtins.open", mock_open(read_data=mock_csv_content)):
        with patch("csv.DictReader", return_value=mock_reader):
            plotting_tools.process_csv()

    assert plotting_tools.headers == ["Time", "Value1", "Value2"]
    assert len(plotting_tools.matched_headers) == 1
    assert plotting_tools.matched_headers[0][1] == "Time"
    assert len(plotting_tools.data) == 2
    assert (
        plotting_tools.data[0]["Time"] == "1"
        and plotting_tools.data[0]["Value1"] == "10"
    )


def test_plot_data(plotting_tools):
    # Test successful plot generation
    data_success = [
        {"Time": "1", "Value1": "10", "Value2": "20"},
        {"Time": "2", "Value1": "15", "Value2": "25"},
    ]
    headers = ["Time", "Value1", "Value2"]
    matched_headers = [(0, "Time")]

    with patch("matplotlib.pyplot.figure"), patch("matplotlib.pyplot.plot"), patch(
        "matplotlib.pyplot.xlabel"
    ), patch("matplotlib.pyplot.ylabel"), patch("matplotlib.pyplot.title"), patch(
        "matplotlib.pyplot.savefig"
    ), patch(
        "matplotlib.pyplot.close"
    ):
        plotting_tools.data = data_success
        plotting_tools.headers = headers
        plotting_tools.matched_headers = matched_headers
        created_plots = plotting_tools.plot_data()
        assert "timevsvalue1" in created_plots
        assert "timevsvalue2" in created_plots

    # Test failure due to non-numeric data
    data_failure = [
        {"Time": "1", "Value1": "A", "Value2": "B"},
        {"Time": "2", "Value1": "C", "Value2": "D"},
    ]

    plotting_tools.data = data_failure
    plotting_tools.headers = headers
    plotting_tools.matched_headers = matched_headers

    with pytest.raises(Exception) as excinfo:
        plotting_tools.plot_data()
        assert "All plots failed due to non-numeric data." in str(excinfo.value)


@pytest.mark.skip(reason="molrender is not pip installable")
def test_run_molrender(path_to_cif, vis_fxns):
    result = vis_fxns.run_molrender(path_to_cif)
    assert result == "Visualization created"


def test_find_png(vis_fxns):
    vis_fxns.starting_files = os.listdir(".")
    test_file = "test_image.png"
    with open(test_file, "w") as f:
        f.write("")
    png_files = vis_fxns._find_png()
    assert test_file in png_files
    os.remove(test_file)


def test_create_notebook(path_to_cif, vis_fxns):
    result = vis_fxns.create_notebook(path_to_cif)
    (f"{vis_fxns.path_registry.ckpt_figures}/{path_to_cif.split('.')[0]}_vis.ipynb")
    assert result == "Visualization Complete"
