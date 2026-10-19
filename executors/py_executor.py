"""
Subprocess-isolated Python executor.

Every test / evaluation runs in its own `python -I` child process with a hard
timeout, so candidate code can:
  * loop forever        -> child is killed, no leaked thread
  * call os._exit / sys.exit -> only the child dies, and it counts as a FAIL
  * define globals      -> nothing leaks into the next task or the harness
"""
import ast
import json
import os
import secrets
import subprocess
import sys
import tempfile
from typing import List

from .executor_types import ExecuteResult, Executor

# Runs one assert against the candidate; prints a tagged JSON result line.
_TEST_HARNESS = r'''
import json, sys, ast, os
payload = json.load(open(sys.argv[1]))
code, test = payload["code"], payload["test"]
ns = {}
def out(ok, output=""):
    print("\n@@RESULT@@" + json.dumps({"ok": ok, "output": output}), flush=True)
    os._exit(0)
try:
    exec("from typing import *\n" + code, ns)
except BaseException as e:
    out(False, f"code failed to load: {type(e).__name__}: {e}")
try:
    exec(test, ns)
    out(True)
except AssertionError:
    try:
        node = ast.parse(test).body[0]
        expr = node.test.left if hasattr(node.test, "left") else node.test
        out(False, repr(eval(compile(ast.Expression(expr), "<t>", "eval"), ns)))
    except BaseException as e:
        out(False, f"{type(e).__name__}: {e}")
except BaseException as e:
    out(False, f"{type(e).__name__}: {e}")
'''

# Runs the full HumanEval check(); success is proven by a random token that the
# candidate cannot know, so os._exit(0) inside the candidate is NOT a pass.
_EVAL_HARNESS = r'''
import sys, os
token = sys.argv[2]
ns = {}
try:
    exec(open(sys.argv[1]).read(), ns)
except BaseException:
    os._exit(1)
print(token, flush=True)
os._exit(0)
'''


def _run_one_test(code: str, test: str, timeout: int, workdir: str) -> dict:
    harness = os.path.join(workdir, "test_harness.py")
    if not os.path.exists(harness):
        with open(harness, "w") as f:
            f.write(_TEST_HARNESS)
    payload = os.path.join(workdir, "payload.json")
    with open(payload, "w") as f:
        json.dump({"code": code, "test": test}, f)
    try:
        p = subprocess.run([sys.executable, "-I", harness, payload],
                           capture_output=True, text=True,
                           timeout=timeout, cwd=workdir)
        lines = [l for l in p.stdout.splitlines() if l.startswith("@@RESULT@@")]
        if lines:
            return json.loads(lines[-1][len("@@RESULT@@"):])
        return {"ok": False, "output": (p.stderr or "no output")[-300:]}
    except subprocess.TimeoutExpired:
        return {"ok": False, "output": "TIMEOUT"}


class PyExecutor(Executor):
    def execute(self, func: str, tests: List[str], timeout: int = 5) -> ExecuteResult:
        success_tests, failed_tests, state = [], [], []
        with tempfile.TemporaryDirectory() as d:
            for t in tests:
                r = _run_one_test(func, t, timeout, d)
                if r["ok"]:
                    success_tests.append(t)
                else:
                    failed_tests.append(f"{t} # output: {r['output']}")
                state.append(bool(r["ok"]))

        feedback = "Tested passed:"
        for t in success_tests:
            feedback += f"\n{t}"
        feedback += "\n\nTests failed:"
        for t in failed_tests:
            feedback += f"\n{t}"
        return ExecuteResult(not failed_tests, feedback, tuple(state))

    def evaluate(self, name: str, func: str, test: str, timeout: int = 5) -> bool:
        """Run the benchmark's full `check(candidate)` in an isolated process."""
        code = f"from typing import *\n{func}\n\n{test}\n\ncheck({name})\n"
        token = secrets.token_hex(16)
        with tempfile.TemporaryDirectory() as d:
            harness = os.path.join(d, "eval_harness.py")
            target = os.path.join(d, "candidate.py")
            with open(harness, "w") as f:
                f.write(_EVAL_HARNESS)
            with open(target, "w") as f:
                f.write(code)
            try:
                p = subprocess.run([sys.executable, "-I", harness, target, token],
                                   capture_output=True, text=True,
                                   timeout=timeout, cwd=d)
            except subprocess.TimeoutExpired:
                return False
            return token in p.stdout


if __name__ == "__main__":
    func = "def add(a, b):\n    while True:\n        x = 1\n    return a + b"
    tests = ["assert add(1, 2) == 3", "assert add(1, 2) == 4"]
    print(PyExecutor().execute(func, tests, timeout=1))
