import asyncio
import contextlib

# import contextlib
import json
import os
from typing import Any, Literal

from analysis_tools import (
    compute_bond_angles,
    compute_contacts,
    compute_distance,
    compute_hbonds,
    compute_rdf,
    compute_rmsd,
    compute_rmsf,
)
from aviary.core import (
    Environment,
    Message,
    Messages,
    Tool,
    ToolRequestMessage,
    ToolResponseMessage,
)
from preprocess_tools import clean_pdb_file, download_pdb_file
from pydantic import BaseModel, ConfigDict, Field
from simulation_tools import setup_and_run_simulation
from state import MDCrowState

from ldp.agent import Agent
from ldp.graph import LLMCallOp, OpResult, compute_graph

os.environ["OPENAI_API_KEY"] = ""


class MySimpleAgent(BaseModel, Agent[MDCrowState]):
    """Simple agent that can invoke tools with a language model."""

    llm_model: dict[str, Any] = Field(
        default={
            "name": "gpt-4o-2024-08-06",
            "temperature": 0.1,
        },
        description="Configuration for the LLM object.",
    )
    guidelines_msg: Message = Field(
        default=Message(role="system", content=""),
        description="Initial guidelines to be shown to the LLM.",
    )

    def __init__(self, guidelines=" ", **kwargs):
        super().__init__(**kwargs)
        self.guidelines_msg = Message(role="system", content="Guidelines:" + guidelines)

        # Create a Op that the agent will use to call the LLM API
        self._llm_call_op = LLMCallOp()

    async def init_state(self, tools: list[Tool], path_registry=None) -> MDCrowState:
        return MDCrowState(tools=tools, path_registry=path_registry)

    @compute_graph()
    async def get_asv(
        self, agent_state: MDCrowState, obs: list[Message]
    ) -> tuple[OpResult[ToolRequestMessage], MDCrowState, float]:
        # Obtain the next agent state, given the environment observation
        next_state = agent_state.get_next_state(obs)
        result = await self._llm_call_op(
            self.llm_model,
            msgs=[
                # We prepend the system guidelines here!
                self.guidelines_msg,
                *next_state.messages,
            ],
            tools=next_state.tools,
        )

        # Extend the the agent state with the new ToolRequestMessage
        next_state.messages = [*next_state.messages, result.value]

        # Agent returns an OpResult, the next agent state and the value, which we set to 0.0
        return result, next_state, 0.0


class MDCrowEnvConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    correct_reward: float = 1.0
    incorrect_reward: float = 0.0
    tool_failure_reward: float = -1.0
    tool_success_reward: float = 0.0
    rel_tol: float = 1e-4

    done_on_failure: bool = True


class MDCrowEnv(Environment[None]):
    def __init__(
        self,
        problem_id: str,
        problem: str,
        answer: float,
        config: MDCrowEnvConfig | None = None,
    ):
        # The problem is not part of the state because it is always the same.
        # Putting it in the state would imply it is somehow affected by .step()
        # or re-initialized by .reset().
        self.problem_id = problem_id
        self.problem = problem
        self.answer = float(answer)  # If passed in as a 0d tensor  # noqa: FURB123

        self.config = config if config is not None else MDCrowEnvConfig()

    @classmethod
    def from_task(cls, task: str) -> "MDCrowEnv":
        return cls(problem_id="task", problem=task, answer=0.0)

    async def reset(self) -> tuple[Messages, list[Tool]]:
        self.state = None  # this environment is effectively stateless
        self.tools = [
            # preprocess tools
            Tool.from_function(download_pdb_file),
            Tool.from_function(clean_pdb_file),
            # simulation tools
            Tool.from_function(setup_and_run_simulation),
            # analysis tools
            Tool.from_function(compute_rmsd),
            Tool.from_function(compute_rmsf),
            Tool.from_function(compute_rdf),
            Tool.from_function(compute_bond_angles),
            Tool.from_function(compute_contacts),
            Tool.from_function(compute_distance),
            Tool.from_function(compute_hbonds),
            # submit answer
            Tool.from_function(self.submit_answer),
        ]
        return [Message(content=self.problem)], self.tools

    async def step(
        self, action: ToolRequestMessage, state: MDCrowState
    ) -> tuple[Messages, float, bool, bool]:
        if not action.tool_calls:
            return (
                [
                    Message(
                        content=(
                            "Must call one of the provided tools"
                            # preprocess tools
                            f" ({download_pdb_file.__name__} or"
                            f" {clean_pdb_file.__name__} or"
                            # simulation tools
                            f" {setup_and_run_simulation.__name__})."
                            # analysis tools
                            f" {compute_rmsd.__name__}) or"
                            f" {compute_rmsf.__name__}) or"
                            f" {compute_rdf.__name__}) or"
                            f" {compute_bond_angles.__name__}) or"
                            f" {compute_contacts.__name__}) or"
                            f" {compute_distance.__name__}) or"
                            f" {compute_hbonds.__name__}) or"
                            # submit answer
                            f" {self.submit_answer.__name__})."
                        )
                    )
                ],
                self.config.tool_failure_reward,
                self.config.done_on_failure,
                False,
            )

        valid_action, invalid_action = self.filter_invalid_tool_calls(action)

        invalid_response_msgs = [
            ToolResponseMessage.from_call(tool_call, content="")
            for tool_call in invalid_action.tool_calls
        ]

        if valid_action.tool_calls:
            # TODO: Just let exec_tool_calls handle invalid tool calls
            # once someone can take a closer look at what response, reward, done
            # would be in that case.
            results = await self.exec_tool_calls(
                valid_action, state=state, handle_invalid_tool_calls=False
            )
            response_msgs = []
            total_reward = 0.0
            any_done = False

            for tool_call, result in zip(valid_action.tool_calls, results, strict=True):
                response, reward, done = json.loads(result.content)

                response_msgs.append(
                    ToolResponseMessage.from_call(tool_call, content=str(response))
                )

                total_reward += reward
                any_done |= done

            return (  # type: ignore[return-value]
                response_msgs + invalid_response_msgs,
                total_reward,
                any_done,
                False,
            )

        return (  # type: ignore[return-value]
            invalid_response_msgs,
            self.config.tool_failure_reward * len(invalid_response_msgs),
            self.config.done_on_failure,
            False,
        )

    def submit_answer(self, answer: str) -> tuple[bool, float, Literal[True]]:
        """Submit the proposed answer and check if it is correct. This action is terminal.

        Args:
            answer: Proposed answer.

        Returns:
            Three-tuple of if correct, associated reward (correct_reward if correct,
                tool_failure_reward if tool failure, otherwise incorrect_reward), and
                True indicating done.
        """
        try:
            correct: bool = (
                abs(float(answer) - self.answer)
                / (abs(self.answer) + self.config.rel_tol)
                < self.config.rel_tol
            )
            reward = (
                self.config.correct_reward if correct else self.config.incorrect_reward
            )
        except ValueError:
            return False, self.config.tool_failure_reward, True
        else:
            return correct, reward, True

    def calculator(self, expr: str) -> tuple[float | str, float, bool]:
        """Calculate a mathematical expression.

        Args:
            expr: A valid Python expression.

        Returns:
            A three-tuple where the first element is the float evaluation if successful,
                or a string containing the failure cause if unsuccessful, the second
                element is the reward associated with success or failure, and the third
                element is a boolean indicating if this action is terminal.
        """
        try:
            expr = expr.strip()
            result = eval(expr)  # noqa: S307  # pylint: disable=eval-used
            with contextlib.suppress(ValueError):  # If possible, downcast float to int
                if int(result) == result:
                    result = int(result)
        except Exception as exc:
            return (
                f"Error using calculator: {exc!r}.",
                self.config.tool_failure_reward,
                self.config.done_on_failure,
            )
        return result, self.config.tool_success_reward, False


env = MDCrowEnv.from_task(
    "Download 1PGA structure from PDB clean, simulate and compute angle distances."
)
agent = MySimpleAgent()


async def main(idx: int = 0):
    # env = GSM8kDataset(split="train").get_new_env_by_idx(idx)
    # agent = MySimpleAgent()

    # Get initial question, available tools from the environment
    obs, tools = await env.reset()
    print(f"Question: {obs[0].content}")

    # Get initial agent state
    agent_state = await agent.init_state(tools=tools, path_registry=None)
    # print("\n",agent_state.messages)
    step = 1
    done = False
    while not done:
        action, agent_state, _ = await agent.get_asv(agent_state, obs)
        # print("\n",agent_state.messages)
        obs, reward, done, _ = await env.step(action.value, agent_state)
        print(
            f"Step {step} - {print_action_obs(action, obs)}, environment reward {reward}"
        )
        step += 1
    await agent.get_asv(agent_state, obs)
    print("Finished! \n")
    return agent_state


def print_action_obs(action: ToolRequestMessage, obs: list[ToolResponseMessage]):
    tool_calls = action.value.tool_calls
    msg = ""
    for tool_call, tool_answer in zip(tool_calls, obs, strict=True):
        tool_name = tool_call.function.name
        tool_args = tool_call.function.arguments
        msg += f"agent action: {tool_name}({tool_args}), environment answer: {tool_answer.content} "
    return msg


async def run_main():
    for i in range(1):
        await main(i)
        # print(agent_state)


if __name__ == "__main__":
    asyncio.run(run_main())
