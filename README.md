MDCrow is a LLM-agent based toolset for Molecular Dynamics.
It's built using Langchain and uses a collection of tools to set up and execute molecular dynamics simulations, particularly in OpenMM.


## Environment Setup
To use the OpenMM features in the agent, please set up a conda environment, following these steps.
```
conda env create -n mdcrow -f environment.yaml
conda activate mdcrow
```

If you already have a conda environment, you can install dependencies before you activate it with the following step.
- Install the necessary conda dependencies: `conda env update -n <YOUR_CONDA_ENV_HERE> -f environment.yaml`



## Installation
```
pip install git+https://github.com/ur-whitelab/MDCrow.git
```

## Usage
The next step is to set up your API keys in your environment. An API key for LLM provider is necessary for this project. Supported LLM providers are OpenAI, TogetherAI, Fireworks, and Anthropic.
Other tools require API keys, such as paper-qa for literature searches. We recommend setting up the keys in a .env file. You can use the provided .env.example file as a template.
1. Copy the `.env.example` file and rename it to `.env`: `cp .env.example .env`
2. Replace the placeholder values in `.env` with your actual keys

You can ask MDCrow to conduct molecular dynamics tasks using OpenAI's GPT model
```
from mdcrow import MDCrow

agent = MDCrow(model="gpt-3.5-turbo")
agent.run("Simulate protein 1ZNI at 300 K for 0.1 ps and calculate the RMSD over time.")
```
Note: to distinguish Together models from the rest, you'll need to add "together\" prefix in model flag, such as `agent = MDCrow(model="together/meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo")`

## LLM Providers
By default, we support LLMs through OpenAI API. However, feel free to use other LLM providers. Make sure to install the necessary package for it. Here's list of packages required for alternative LLM providers we support:
- `pip install langchain-together` to use models from TogetherAI
- `pip install langchain-anthropic` to use models from Anthropic
- `pip install langchain-fireworks` to use models from Fireworks


## Contributing

We welcome contributions to MDCrow! If you're interested in contributing to the project, please check out our [Contributor's Guide](CONTRIBUTING.md) for detailed instructions on getting started, feature development, and the pull request process.

We value and appreciate all contributions to MDCrow.
