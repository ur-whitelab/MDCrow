from typing import Optional

import nest_asyncio
import paperqa
from langchain.base_language import BaseLanguageModel
from langchain.tools import BaseTool

from mdcrow.utils import PathRegistry


def scholar2result_llm(llm, query, path_registry):
    paper_directory = path_registry.ckpt_papers
    if paper_directory is None:
        raise ValueError(
            "'paper_dir' is None. To use this tool, the user "
            "must provide a directory with PDFs at the start."
        )
    print("Paper Directory", paper_directory)
    llm_name = llm.model_name
    if llm_name.startswith("gpt") or llm_name.startswith("claude"):
        settings = paperqa.Settings(
            llm=llm_name,
            summary_llm=llm_name,
            temperature=llm.temperature,
            paper_directory=paper_directory,
        )
    else:
        settings = paperqa.Settings(
            temperature=llm.temperature,  # uses default gpt model in paperqa
            paper_directory=paper_directory,
        )
    response = paperqa.ask(query, settings=settings)
    answer = response.answer.formatted_answer
    if "I cannot answer." in answer:
        answer += f" Check to ensure there's papers in {paper_directory}"
    print(answer)
    return answer


class Scholar2ResultLLM(BaseTool):
    name = "LiteratureSearch"
    description = (
        "Useful to answer questions that may be found in literature. "
        "Ask a specific question as the input."
    )
    llm: BaseLanguageModel = None
    path_registry: Optional[PathRegistry]

    def __init__(self, llm, path_registry):
        super().__init__()
        self.llm = llm
        self.path_registry = path_registry

    def _run(self, query) -> str:
        nest_asyncio.apply()
        try:
            return scholar2result_llm(self.llm, query, self.path_registry)
        except Exception as e:
            print(e)
            return f"Failed. {type(e).__name__}: {e}"

    async def _arun(self, query) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("this tool does not support async")
