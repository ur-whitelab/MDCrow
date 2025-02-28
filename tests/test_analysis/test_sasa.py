import warnings
from unittest.mock import patch

import numpy as np
import pytest

from mdcrow.tools.base_tools.analysis_tools.sasa import (
    SASAFunctions,
    SolventAccessibleSurfaceArea,
)


@pytest.fixture
def get_sasa_functions_with_files(get_registry):
    registry = get_registry("raw", True)
    traj_fileid = "rec0_butane_123456"
    top_fileid = "top_sim0_butane_123456"
    return SASAFunctions(registry, top_fileid, traj_fileid)


def test_sasa_analysis_init_success(get_registry):
    registry = get_registry("raw", True)
    with patch.object(
        registry, "get_mapped_path", wraps=registry.get_mapped_path
    ) as mocked_get_mapped_path:
        analysis = SASAFunctions(
            registry, "top_sim0_butane_123456", "rec0_butane_123456"
        )
        assert mocked_get_mapped_path.call_count == 2
        assert analysis.path_registry == registry
        assert analysis.molecule_name == "sim0_butane_123456"
        assert analysis.traj is not None


def test_sasa_analysis_init_success_no_traj(get_registry):
    registry = get_registry("raw", True)
    with patch.object(
        registry, "get_mapped_path", wraps=registry.get_mapped_path
    ) as mocked_get_mapped_path:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            analysis = SASAFunctions(
                registry, "top_sim0_butane_123456", mol_name="butane"
            )
        mocked_get_mapped_path.assert_called_once()
        assert analysis.path_registry == registry
        assert analysis.molecule_name == "butane"


def test_sasa_tool_init(get_registry):
    registry = get_registry("raw", False)
    tool = SolventAccessibleSurfaceArea(path_registry=registry)
    assert tool.name == "SolventAccessibleSurfaceArea"
    assert tool.path_registry == registry


@patch("os.path.exists", return_value=False)
@patch("mdcrow.tools.base_tools.analysis_tools.sasa.np.savetxt")
def test_calculate_sasa(mock_savetxt, mock_exists, get_sasa_functions_with_files):
    analysis = get_sasa_functions_with_files
    result = analysis.calculate_sasa()
    assert "SASA values computed and saved" in result
    mock_savetxt.assert_called_once()
    assert analysis.residue_sasa is not None
    assert analysis.total_sasa is not None


@patch("mdcrow.tools.base_tools.analysis_tools.sasa.plt.savefig")
@patch("mdcrow.tools.base_tools.analysis_tools.sasa.plt.close")
def test_plot_sasa(mock_close, mock_savefig, get_sasa_functions_with_files):
    analysis = get_sasa_functions_with_files
    analysis.residue_sasa = np.array([[1, 2], [3, 4]])
    analysis.total_sasa = np.array([1, 2])
    with patch.object(SASAFunctions, "calculate_sasa") as mock_calc:
        result = analysis.plot_sasa()
        mock_calc.assert_not_called()
        mock_savefig.assert_called_once()
        assert "SASA analysis completed" in result
