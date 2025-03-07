from typing import Self

from aviary.core import Message, Tool
from pydantic import BaseModel, ConfigDict, Field
from utils import PathRegistry


class MDCrowState(BaseModel):
    # model_config = ConfigDict(extra="forbid")
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")
    ckpt_dir: str = "ckpt"
    reward: float = 0.0
    steps: int = 0
    done: bool = False
    path_registry: PathRegistry = None

    tools: list[Tool] = Field(default_factory=list)
    messages: list[Message] = Field(default_factory=list)

    def __init__(
        self, tools, path_registry=None, ckpt_dir="ckpt", paper_dir=None, messages=None
    ):
        super().__init__()

        if path_registry is None:
            self.path_registry = PathRegistry.get_instance(ckpt_dir)
            self.ckpt_dir = self.path_registry.ckpt_dir
        else:
            self.path_registry = path_registry
            self.ckpt_dir = path_registry.ckpt_dir
        self.tools = tools
        self.messages = messages or []

    def get_next_state(
        self,
        obs: list[Message] | None = None,
    ) -> self:
        """
        Return the next agent state based on current state and optional messages.

        Args:
            obs: Optional observation messages to use in creating the next state.

        Returns:
            The next agent state.
        """
        return type(self)(
            tools=self.tools,
            messages=self.messages + (obs or []),
            path_registry=self.path_registry,
        )
