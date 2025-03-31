import paperqa

llm_model_args = {
    "name": "gpt-4o-2024-08-06",
    "temperature": 0.5,
}


def scholar2result_llm(state, query):
    """
     Useful to answer questions that may be found in literature.
    Ask a specific question as the input.

    Args:
        state (MDCrowState): The state of the MDCrow environment.
        query (str): The question to ask the model.
    """

    path_registry = state.path_registry
    paper_directory = path_registry.ckpt_papers
    if paper_directory is None:
        raise ValueError(
            "'paper_dir' is None. To use this tool, the user "
            "must provide a directory with PDFs at the start."
        )
    print("Paper Directory", paper_directory)
    llm_name = llm_model_args["model_name"]
    temperature = llm_model_args["temperature"]
    if llm_name.startswith("gpt") or llm_name.startswith("claude"):
        settings = paperqa.Settings(
            llm=llm_name,
            summary_llm=llm_name,
            temperature=temperature,
            paper_directory=paper_directory,
        )
    else:
        settings = paperqa.Settings(
            temperature=temperature,  # uses default gpt model in paperqa
            paper_directory=paper_directory,
        )
    try:
        response = paperqa.ask(query, settings=settings)
        answer = response.answer.formatted_answer
        if "I cannot answer." in answer:
            answer += f" Check to ensure there's papers in {paper_directory}"
        print(answer)
        return answer, 0, False

    except Exception as e:
        print("Error in scholar2result_llm:", e)
        return (
            f"Failed. Error in scholar2result_llm: {e}",
            0,
            False,
        )
