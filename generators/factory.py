from .py_generate import PyGenerator
from .generator_types import Generator
from .model import ModelBase, GPT4, GPT35, GPTDavinci
from .gemini_client import GeminiModel


def generator_factory(lang: str) -> Generator:
    if lang == "py" or lang == "python":
        return PyGenerator()
    else:
        raise ValueError(f"Invalid language for generator: {lang}"
                          " (this trimmed project supports Python only — "
                          "see the original repo for Rust support)")


def model_factory(model_name: str) -> ModelBase:
    if model_name.startswith("gemini"):
        return GeminiModel(model_name=model_name)
    elif model_name == "gpt-4":
        return GPT4()
    elif model_name == "gpt-3.5-turbo":
        return GPT35()
    elif model_name.startswith("text-davinci"):
        return GPTDavinci(model_name)
    else:
        raise ValueError(f"Invalid model name: {model_name}")
