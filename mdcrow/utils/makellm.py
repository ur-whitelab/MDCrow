import importlib.util

from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


def check_package_exists(package_name, model):
    if not importlib.util.find_spec(package_name):
        raise ImportError(
            f"The package required to run model '{model}' is missing: '{package_name}'."
        )


def _make_llm(model, temp, streaming):
    if model.startswith("gpt-3.5-turbo") or model.startswith("gpt-4"):
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            temperature=temp,
            model_name=model,
            request_timeout=1000,
            streaming=streaming,
            callbacks=[StreamingStdOutCallbackHandler()] if streaming else None,
        )
    elif model.startswith("accounts/fireworks"):
        check_package_exists("langchain_fireworks", model)
        from langchain_fireworks import ChatFireworks

        llm = ChatFireworks(
            temperature=temp,
            model_name=model,
            request_timeout=1000,
            streaming=streaming,
            callbacks=[StreamingStdOutCallbackHandler()] if streaming else None,
        )
    elif model.startswith("together/"):
        # user needs to add 'together/' prefix to use TogetherAI provider
        check_package_exists("langchain_together", model)
        from langchain_together import ChatTogether

        llm = ChatTogether(
            temperature=temp,
            model=model.replace("together/", ""),
            request_timeout=1000,
            streaming=streaming,
            callbacks=[StreamingStdOutCallbackHandler()] if streaming else None,
        )
    elif model.startswith("claude"):
        check_package_exists("langchain_anthropic", model)
        from langchain_anthropic import ChatAnthropic

        llm = ChatAnthropic(
            temperature=temp,
            model_name=model,
            streaming=streaming,
            callbacks=[StreamingStdOutCallbackHandler()] if streaming else None,
        )
    else:
        raise ValueError(f"Unrecognized or unsupported model name: {model}")
    return llm
