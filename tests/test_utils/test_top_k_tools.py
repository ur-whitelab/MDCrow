import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from mdcrow.tools.maketools import get_relevant_tools


@pytest.fixture
def mock_llm():
    return MagicMock()


@pytest.fixture
def mock_tools():
    Tool = MagicMock()
    tool1 = Tool(name="Tool1", description="This is the first tool")
    tool2 = Tool(name="Tool2", description="This is the second tool")
    tool3 = Tool(name="Tool3", description="This is the third tool")
    return [tool1, tool2, tool3]


@patch("mdcrow.tools.maketools.make_all_tools")
@patch("mdcrow.tools.maketools.OpenAIEmbeddings")
def test_get_relevant_tools_with_openai_embeddings(
    mock_openai_embeddings, mock_make_all_tools, mock_llm, mock_tools
):
    mock_make_all_tools.return_value = mock_tools
    mock_embed_documents = mock_openai_embeddings.return_value.embed_documents
    mock_embed_query = mock_openai_embeddings.return_value.embed_query
    mock_embed_documents.return_value = np.random.rand(3, 512)
    mock_embed_query.return_value = np.random.rand(512)

    with patch.dict(
        os.environ, {"OPENAI_API_KEY": "test_key"}  # pragma: allowlist secret
    ):
        relevant_tools = get_relevant_tools("test query", mock_llm, top_k_tools=2)
        assert len(relevant_tools) == 2
        assert relevant_tools[0] in mock_tools
        assert relevant_tools[1] in mock_tools


@patch("mdcrow.tools.maketools.make_all_tools")
@patch("mdcrow.tools.maketools.TfidfVectorizer")
def test_get_relevant_tools_with_tfidf(
    mock_tfidf_vectorizer, mock_make_all_tools, mock_llm, mock_tools
):
    mock_make_all_tools.return_value = mock_tools
    mock_vectorizer = mock_tfidf_vectorizer.return_value
    mock_vectorizer.fit_transform.return_value = np.random.rand(3, 10)
    mock_vectorizer.transform.return_value = np.random.rand(1, 10)

    with patch.dict(os.environ, {}, clear=True):  # ensure OPENAI_API_KEY is not set
        relevant_tools = get_relevant_tools("test query", mock_llm, top_k_tools=2)
        assert len(relevant_tools) == 2
        assert relevant_tools[0] in mock_tools
        assert relevant_tools[1] in mock_tools


@patch("mdcrow.tools.maketools.make_all_tools")
def test_get_relevant_tools_with_no_tools(mock_make_all_tools, mock_llm):
    mock_make_all_tools.return_value = []

    with patch.dict(os.environ, {}, clear=True):
        relevant_tools = get_relevant_tools("test query", mock_llm)
        assert relevant_tools is None


@patch("mdcrow.tools.maketools.make_all_tools")
@patch("mdcrow.tools.maketools.OpenAIEmbeddings")
def test_get_relevant_tools_with_openai_exception(
    mock_openai_embeddings, mock_make_all_tools, mock_llm, mock_tools
):
    mock_make_all_tools.return_value = mock_tools
    mock_embed_documents = mock_openai_embeddings.return_value.embed_documents
    mock_embed_documents.side_effect = Exception("Embedding error")

    with patch.dict(
        os.environ, {"OPENAI_API_KEY": "test_key"}  # pragma: allowlist secret
    ):
        relevant_tools = get_relevant_tools("test query", mock_llm)
        assert relevant_tools is None


@patch("mdcrow.tools.maketools.make_all_tools")
def test_get_relevant_tools_top_k(mock_make_all_tools, mock_llm, mock_tools):
    mock_make_all_tools.return_value = mock_tools

    with patch.dict(os.environ, {}, clear=True):
        relevant_tools = get_relevant_tools("test query", mock_llm, top_k_tools=1)
        assert len(relevant_tools) == 1
        assert relevant_tools[0] in mock_tools

        relevant_tools = get_relevant_tools("test query", mock_llm, top_k_tools=5)
        assert len(relevant_tools) == len(mock_tools)
