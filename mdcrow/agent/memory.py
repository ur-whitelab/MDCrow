import json
import os
import random
import string

from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from mdcrow.utils import PathRegistry

agent_summary_template = PromptTemplate(
    input_variables=["agent_trace"],
    template="""
    Your job is to summarize the
    history of an agent's attempts to
    solve a problem. Be sure to
    include the prompt, every step,
    and the final
    solution. You may include other
    details as well, if you think
    they are relevant.

    Here is the history:
        {agent_trace}
        """,
)


class MemoryManager:
    def __init__(
        self,
        path_registry: PathRegistry,
        llm,
        run_id="",
    ):
        self.path_registry = path_registry
        self.dir_name = f"{path_registry.ckpt_memory}"
        self.run_id = run_id
        if not self.run_id:
            self.run_id = self.new_run_id()
            pull_mem = False
            self.run_id_mem = None
        else:
            pull_mem = True

        self.llm_agent_trace = agent_summary_template | llm | StrOutputParser()

        self._make_all_dirs()
        if pull_mem:
            self.run_id_mem = self.pull_agent_summary_from_mem(self.run_id)

    def _make_all_dirs(self):
        """
        This function creates the directories for the memory manager.

        Parameters:
        - None

        Returns:
        - None
        """
        # add any history dir here
        self.agent_trace_summary = f"{self.dir_name}/agent_run_summaries.json"

    def new_run_id(self) -> str:
        """
        This function generates a new run_id.

        Parameters:
        - None

        Returns:
        - str: The new run_id.
        """
        length = 8
        characters = string.ascii_uppercase + string.digits
        return "".join(random.choice(characters) for _ in range(length))

    def _write_to_json(self, data, path):
        """
        This function writes data to a json file.

        Parameters:
        - data (dict): The data to write to the json file.
        - path (str): The path to the json file.

        Returns:
        - None
        """
        if os.path.exists(path):
            with open(path, "r") as f:
                data_original = json.load(f)
            data_original.update(data)
        else:
            data_original = data
        with open(path, "w") as f:
            f.write(json.dumps(data_original, indent=4, sort_keys=True))

    def get_summary_number(self):
        """
        Retrieves the next number for the agent summary.

        Parameters:
        - None

        Returns:
        - str: The next summary number.
        If no previous history exists or the history file is empty,
        it returns "0".
        """
        if not os.path.exists(self.agent_trace_summary):
            return str(0)
        with open(self.agent_trace_summary, "r") as f:
            summary = json.load(f)
        if not summary:
            return str(0)
        return str(len(summary.keys()))

    def generate_agent_summary(self, agent_trace):
        """
        This function generates a summary of the agent's run history
        and writes it to a json file.
        There will be one file per run_id.

        Parameters:
        - agent_trace (dict): The agent's run history.

        Returns:
        - None
        """
        llm_out = self.llm_agent_trace.invoke({"agent_trace": agent_trace})
        key_str = f"{self.run_id}.{self.get_summary_number()}"
        run_summary = {key_str: llm_out}
        self._write_to_json(run_summary, self.agent_trace_summary)
        return None

    def pull_agent_summary_from_mem(self, run_id: str, run_num: int = 0):
        """
        This function pulls the agent summary for a given run
        number from the current run_id's agent summary file.

        Parameters:
        - run_num (int): The run number to pull the summary for.
        Default is 0, which pulls the most recent summary.

        Returns:
        - str: The agent summary for the given run number.
        If the summary does not exist, it returns None.
        """
        run_id_full = f"{run_id}.{run_num}"
        if not os.path.exists(self.agent_trace_summary):
            return "Path does not exist."
        with open(self.agent_trace_summary, "r") as f:
            summary = json.load(f)
        if run_id_full not in summary.keys():
            return "Run ID not found. Keys are: " + str(summary.keys())
        return summary[run_id_full]
