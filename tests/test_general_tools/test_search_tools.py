import pytest

from mdcrow.ldp_env.analysis_tools.scholar import scholar2result_llm
from mdcrow.ldp_env.state import MDCrowState


@pytest.fixture
def question():
    return "What are the effects of norhalichondrin B in mammals?"


@pytest.mark.skip(reason="Requires actual API call and local PDFs in paper directory.")
def test_scholar2result_llm(question, get_registry):
    state = MDCrowState(path_registry=get_registry("raw", False))
    result, code, stop_flag = scholar2result_llm(state, question)

    assert isinstance(result, str)
    assert len(result) > 0
    assert code == 0
    assert stop_flag is False
