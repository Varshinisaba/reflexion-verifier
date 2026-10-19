from .py_executor import PyExecutor
from .executor_types import Executor


def executor_factory(lang: str, is_leet: bool = False) -> Executor:
    if lang == "py" or lang == "python":
        if is_leet:
            raise NotImplementedError(
                "LeetCode execution isn't included in this trimmed project — "
                "see the original noahshinn/reflexion repo's leet_executor.py "
                "if you need it."
            )
        return PyExecutor()
    else:
        raise ValueError(f"Invalid language for executor: {lang}"
                          " (this trimmed project supports Python only)")
