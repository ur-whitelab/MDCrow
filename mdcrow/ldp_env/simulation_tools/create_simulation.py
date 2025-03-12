import os
import textwrap

from aviary.core import Message
from state import MDCrowState
from utils import FileType

from ldp.graph import LLMCallOp


class ModifyScriptUtils:

    def _prompt_summary(self, task: dict):

        prompt_template = (
            "You're an expert in programming and in molecular dynamics. "
            "Your job is to make a script to make a simulation "
            "in openmm. "
            "Your starting point is a base script that runs a protein on its own. "
            "The protein itself doesn't require more preparation. "
            "The forcefields, integrator, and constraints are already set up for you. "
            "You need to add lines to fulfill the user requirement. "
            "Your answer has to be the modified script. "
            "Your answer should be a python script. "
            "Don't use ''' to comment out the code; use # instead. "
            "Describe your thoughts and changes before you start writing the script. "
            "The script will be rum as it is, so make it completely. "
            "The format should be as follows: "
            "THOUGHTS: (Your thoughts as an openmm expert with the base "
            "script and the query) \n"
            "CHANGES:(what modifications you're doing to the script)\n "
            "SCRIPT: (The COMPLETE modified script)\n "
            "FINAL THOUGHTS: (Optional, Any final thoughts or comments\n "
            "you have about the script\n "
            "Base_SCRIPT:\n"
            "{base_script} \n"
            "Question: {query} "
        )

        mssg = Message(role="system", content=prompt_template.format(**task))

        return mssg

    def remove_leading_spaces(self, text):
        lines = text.split("\n")
        stripped_lines = [line.lstrip() for line in lines]
        return "\n".join(stripped_lines)


llm_model_args = {
    "name": "gpt-4o-2024-08-06",
    "temperature": 0.5,
}

llm_call_op = LLMCallOp()


def llm_call(message):
    response = llm_call_op(llm_model_args, message)
    return response.values.content


def modify_simulation_script(
    state: MDCrowState, script_id: str, query: str, modifysim_no_run: bool = False
):
    """
     This tool takes a base simulation script and a user requirement and \
          returns a modified script.

    Args:
        script_id (str): File ID of the simulation script to be modified.
        query (str): Detailed description of the simulation requirements.
            - Should include details such as force fields, integrator type, constraints,
              and any other relevant parameters.
            - Also mention the specific protein being simulated.
        modifysim_no_run (bool): If True, the modified script will not be run.
            - Default is False.


    """
    path_registry = state.path_registry
    if not path_registry:
        return "Failed. No path registry provided", 0, False  # this should not happen
    if not script_id:
        return (
            (
                "Failed. No id provided. The keys for the input are: "
                "query' and 'script_id'"
            ),
            0,
            False,
        )
    current_ids = path_registry.list_path_names()
    if script_id not in current_ids:
        return (
            (
                f"Failed. File ID not found: {script_id}, make sure "
                "the script ID is correct"
            ),
            0,
            False,
        )
    try:
        base_script_path = path_registry.get_mapped_path(script_id)
        parts = base_script_path.split("/")
        if len(parts) > 1:
            parts[-1]
    except Exception as e:
        return f"Failed. Error getting path from file id: {e}", 0, False
    if os.path.exists(base_script_path):
        with open(base_script_path, "r") as file:
            base_script = file.read()
    else:
        return f"Failed. File not found: {script_id}", 0, False

    base_script = "".join(base_script)
    utils = ModifyScriptUtils()

    description = query
    message = utils._prompt_summary(
        task={"base_script": base_script, "query": description}
    )

    # Get the response from the LLM
    answer = llm_call(message)
    thoughts, new_script = answer.split("SCRIPT:")
    script_content = new_script
    if "FINAL THOUGHTS:" in script_content:
        script_content, final_thoughts = script_content.split("FINAL THOUGHTS:")
    # replace ''' with #
    script_content = script_content.replace("```", "#")
    script_content = textwrap.dedent(script_content).strip()
    # Write to file
    filename = path_registry.write_file_name(
        type=FileType.SIMULATION, Sim_id=script_id, modified=True
    )
    file_id = path_registry.get_fileid(filename, type=FileType.SIMULATION)
    directory = f"{path_registry.ckpt_simulations}"
    with open(f"{directory}/{filename}", "w") as file:
        file.write(script_content)

    path_registry.map_path(file_id, f"{directory}/{filename}", description)
    # if no-run mode is on, return the file id
    if modifysim_no_run:
        return (
            (
                f"Succeeded. Script modified successfully. \
            Modified Script ID: {file_id}"
            ),
            0,
            False,
        )

    # if no-run mode is off, try to run the script
    try:
        exec(script_content)
        return (
            f"Succeeded. Script modified and ran \
            successfully. Modified Script ID: {file_id}",
            0,
            False,
        )
    except Exception as e:
        return (
            (
                f"Failed. Error running the script: {e}."
                "Modified Script ID: {file_id}. If you want to try to correct the "
                "script, use the file id of the modified to correct the script."
            ),
            0,
            False,
        )
