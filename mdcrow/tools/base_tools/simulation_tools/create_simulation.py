import os
import textwrap
from typing import Optional

from langchain.base_language import BaseLanguageModel
from langchain.prompts import PromptTemplate
from langchain.tools import BaseTool
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field

from mdcrow.utils import FileType, PathRegistry


class ModifyScriptUtils:
    llm: Optional[BaseLanguageModel]

    def __init__(self, llm):
        self.llm = llm

    def _prompt_summary(self, task: dict):
        if not self.llm:
            raise ValueError("No language model provided at ModifyScriptTool")

        prompt_template = (
            "You're an expert programmer and in molecular dynamics. "
            "Your job is to make a script to make a simulation "
            "in openmm. "
            "Youre starting point is a base script that runs a protein on its own. "
            "The protein itself doesnt require more preperation. "
            "The forcefields, integrator, and constraints are already set up for you. "
            "You need to add lines to fullfill the user requirement. "
            "Your answer has to be the modified script. "
            "Your answer should be a python script. "
            "Dont use ''' to comment out the code, use # instead. "
            "Describe your thoughts and changes before you start writing the script. "
            "The script will be rum as it is, so make it completely. "
            "The format should be as follows: "
            "THOUGHTS: (Your thoughts as an openmm expert with the base "
            "script and the query) \n"
            "CHANGES:(what modifications youre doing to the script)\n "
            "SCRIPT: (The COMPLETE modified script)\n "
            "FINAL THOUGHTS: (Optional, Any final thoughts or comments\n "
            "you have about the script\n "
            "Base_SCRIPT:\n"
            "{base_script} \n"
            "Question: {query} "
        )

        prompt = PromptTemplate(
            template=prompt_template, input_variables=["base_script", "query"]
        )
        llm_chain = prompt | self.llm | StrOutputParser()

        return llm_chain.invoke(task)

        # Remove leading spaces for proper formatting

    def remove_leading_spaces(self, text):
        lines = text.split("\n")
        stripped_lines = [line.lstrip() for line in lines]
        return "\n".join(stripped_lines)


class ModifyScriptInput(BaseModel):
    script_id: str = Field(..., description=" File ID of the simulation script file")
    query: str = Field(
        ...,
        description=(
            "simulation required by the user. Be as descriptive as possible"
            " including requirements of the simulation, such as the forcefields, "
            "integrator, and constraints. Also, mention the protein you are working on."
            "as on what protein you are working."
        ),
    )


class ModifyBaseSimulationScriptTool(BaseTool):
    name: str = "ModifyScriptTool"
    description: str = (
        "This tool takes a base simulation script and a user "
        "requirement and returns a modified script. "
    )

    args_schema = ModifyScriptInput
    llm: Optional[BaseLanguageModel]
    path_registry: Optional[PathRegistry]
    safe_mode: Optional[bool]

    def __init__(self, path_registry, llm, safe_mode=False):
        super().__init__()
        self.path_registry = path_registry
        self.llm = llm
        self.safe_mode = safe_mode

    def _run(self, script_id: str, query: str) -> str:
        # if len(args) > 0:
        #     return (
        #         "Failed. This tool expects you to provide the input as a "
        #         "dictionary: {'query': 'your query', 'script': 'script id'}"
        #     )
        if not self.path_registry:
            return "Failed. No path registry provided"  # this should not happen
        base_script_id = script_id
        if not base_script_id:
            return (
                "Failed. No id provided. The keys for the input are: "
                "query' and 'script_id'"
            )
        current_ids = self.path_registry.list_path_names()
        if base_script_id not in current_ids:
            return (
                f"Failed. File ID not found: {base_script_id}, make sure "
                "the script ID is correct"
            )
        try:
            base_script_path = self.path_registry.get_mapped_path(base_script_id)
            parts = base_script_path.split("/")
            if len(parts) > 1:
                parts[-1]
        except Exception as e:
            return f"Failed. Error getting path from file id: {e}"
        if os.path.exists(base_script_path):
            with open(base_script_path, "r") as file:
                base_script = file.read()
        else:
            return f"Failed. File not found: {base_script_id}"

        base_script = "".join(base_script)
        utils = ModifyScriptUtils(self.llm)

        description = query
        answer = utils._prompt_summary(
            task={"base_script": base_script, "query": description}
        )
        print("This the answer from the LLM\n\n", answer)
        # script = answer["text"]
        thoughts, new_script = answer.split("SCRIPT:")
        # script_content = utils.remove_leading_spaces(new_script)
        script_content = new_script
        if "FINAL THOUGHTS:" in script_content:
            script_content, final_thoughts = script_content.split("FINAL THOUGHTS:")
        # replace ''' with #
        script_content = script_content.replace("```", "#")
        script_content = textwrap.dedent(script_content).strip()
        # Write to file
        filename = self.path_registry.write_file_name(
            type=FileType.SIMULATION, Sim_id=base_script_id, modified=True
        )
        file_id = self.path_registry.get_fileid(filename, type=FileType.SIMULATION)
        directory = f"{self.path_registry.ckpt_simulations}"
        with open(f"{directory}/{filename}", "w") as file:
            file.write(script_content)

        self.path_registry.map_path(file_id, f"{directory}/{filename}", description)
        # if safe mode is on, return the file id
        if self.safe_mode:
            return f"Succeeded. Script modified successfully. Modified Script ID: {file_id}"
        # if safe mode is off, try to run the script
        try:
            exec(script_content)
            return f"Succeeded. Script modified and ran \
                successfully. Modified Script ID: {file_id}"
        except Exception as e:
            return (
                f"Failed. Error running the script: {e}."
                "Modified Script ID: {file_id}. If you want to try to correct the "
                "script, use the file id of the modified to correct the script."
            )

    async def _arun(self, query) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("this tool does not support async")
