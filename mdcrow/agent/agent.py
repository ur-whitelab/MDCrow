import os
from datetime import datetime

from dotenv import load_dotenv
from langchain.agents import AgentExecutor, OpenAIFunctionsAgent
from langchain.agents.structured_chat.base import StructuredChatAgent

from ..tools import get_relevant_tools, make_all_tools
from ..utils import PathRegistry, SetCheckpoint, _make_llm
from .memory import MemoryManager
from .prompt import openaifxn_prompt, structured_prompt

load_dotenv()


class AgentType:
    valid_models = {
        "Structured": StructuredChatAgent,
        "OpenAIFunctionsAgent": OpenAIFunctionsAgent,
    }

    @classmethod
    def get_agent(cls, model_name: str = "OpenAIFunctionsAgent"):
        try:
            agent = cls.valid_models[model_name]
            return agent
        except KeyError:
            raise ValueError(
                f"""Invalid agent type: {model_name}
                Please choose from {cls.valid_models.keys()}"""
            )


class MDCrow:
    def __init__(
        self,
        tools=None,
        agent_type="OpenAIFunctionsAgent",  # this can also be structured_chat
        model="gpt-4-1106-preview",  # current name for gpt-4 turbo
        tools_model=None,
        temp=0.1,
        streaming=True,
        verbose=False,
        ckpt_dir="ckpt",
        top_k_tools=20,  # set "all" if you want to use all tools
        use_human_tool=False,
        uploaded_files=[],  # user input files to add to path registry
        run_id="",
        use_memory=False,
        safe_mode=False,
        paper_dir=None,  # papers for pqa, relative path within repo
    ):
        self.llm = _make_llm(model, temp, streaming)
        if tools_model is None:
            tools_model = model
        self.tools_llm = _make_llm(tools_model, temp, streaming)

        self.use_memory = use_memory
        self.path_registry = PathRegistry.get_instance(ckpt_dir, paper_dir)
        self.ckpt_dir = self.path_registry.ckpt_dir
        self.memory = MemoryManager(self.path_registry, self.tools_llm, run_id=run_id)
        self.run_id = self.memory.run_id

        self.uploaded_files = uploaded_files
        # for file in uploaded_files:  # todo -> allow users to add descriptions?
        # self.path_registry.map_path(file, file, description="User uploaded file")

        self.agent = None
        self.agent_type = agent_type
        self.top_k_tools = top_k_tools
        self.use_human_tool = use_human_tool
        self.user_tools = tools
        self.verbose = verbose

        if self.uploaded_files:
            self.add_file(self.uploaded_files)
        self.safe_mode = safe_mode

    def _add_single_file(self, file_path, description=None):
        now = datetime.now()
        # Format the date and time as "YYYYMMDD_HHMMSS"
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        i = 0
        ID = "UPL_" + str(i) + timestamp
        while ID in self.path_registry.list_path_names():  # check if ID already exists
            i += 1
            ID = "UPL_" + str(i) + timestamp
        if not description:
            # asks for user input to add description for file file_path
            # wait for 20 seconds or set up a default description
            description = "User uploaded file"
        print(f"Adding file {file_path} with ID {ID}\n")
        self.path_registry.map_path(ID, file_path, description=description)

    def add_file(self, uploaded_files):
        if type(uploaded_files) == str:
            self._add_single_file(uploaded_files)
        elif type(uploaded_files) == tuple:
            self._add_single_file(uploaded_files[0], description=uploaded_files[1])
        elif type(uploaded_files) == list:
            for file_path in uploaded_files:
                print(f"Adding file {file_path}\n")
                print(type(file_path))
                self.add_file(file_path)
        else:
            raise ValueError(
                "Invalid input. Please provide a file path \
                             or list of file paths. Optionally, tuple or list of tuples\
                             of file path and description"
            )

    def _initialize_tools_and_agent(self, user_input=None):
        """Retrieve tools and initialize the agent."""
        if self.user_tools is not None:
            self.tools = self.user_tools
        else:
            if self.top_k_tools != "all" and user_input is not None:
                # retrieve only tools relevant to user input
                self.tools = get_relevant_tools(
                    query=user_input,
                    llm=self.tools_llm,
                    top_k_tools=self.top_k_tools,
                    human=self.use_human_tool,
                )
            else:
                # retrieve all tools, including new tools if any
                self.tools = make_all_tools(
                    self.tools_llm,
                    human=self.use_human_tool,
                    safe_mode=self.safe_mode,
                )
        return AgentExecutor.from_agent_and_tools(
            tools=self.tools,
            agent=AgentType.get_agent(self.agent_type).from_llm_and_tools(
                self.llm,
                self.tools,
            ),
            verbose=self.verbose,
            handle_parsing_errors=True,
        )

    def run(self, user_input, callbacks=None):
        run_memory = self.memory.run_id_mem if self.use_memory else None
        if self.agent_type == "Structured":
            self.prompt = structured_prompt.format(input=user_input, context=run_memory)
        elif self.agent_type == "OpenAIFunctionsAgent":
            self.prompt = openaifxn_prompt.format(input=user_input, context=run_memory)
        self.agent = self._initialize_tools_and_agent(user_input)
        model_output = self.agent.invoke(self.prompt, callbacks=callbacks)
        if self.use_memory:
            self.memory.generate_agent_summary(model_output)
            print("Your run id is: ", self.run_id)
        return model_output, self.run_id

    def iter(self, user_input, include_run_info=True):
        run_memory = self.memory.run_id_mem if self.use_memory else None

        if self.agent is None:
            if self.agent_type == "Structured":
                self.prompt = structured_prompt.format(
                    input=user_input, context=run_memory
                )
            elif self.agent_type == "OpenAIFunctionsAgent":
                self.prompt = openaifxn_prompt.format(
                    input=user_input, context=run_memory
                )
                self.agent = self._initialize_tools_and_agent(user_input)
            for step in self.agent.iter(self.prompt, include_run_info=include_run_info):
                yield step

    def force_clear_mem(self, all=False) -> str:
        if all:
            ckpt_dir = os.path.abspath(os.path.dirname(self.path_registry.ckpt_dir))
        else:
            ckpt_dir = self.path_registry.ckpt_dir
        confirmation = "nonsense"
        while confirmation.lower() not in ["yes", "no"]:
            confirmation = input(
                "Are you sure you want to"
                "clear memory? This will "
                "remove all saved "
                "checkpoints? (yes/no): "
            )

        if confirmation.lower() == "yes":
            set_ckpt = SetCheckpoint()
            set_ckpt.clear_all_ckpts(ckpt_dir)
            return "All checkpoints have been removed."
        else:
            return "Action canceled."
