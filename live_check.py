"""
Live mode: paste code (+ optional tests / description) and run the
Reflexion + Verifier loop on it.

Differences from the batch loop in reflexion.py:
  * Pasted code runs in a SUBPROCESS per test (timeout-safe, can't hang or
    pollute the API process).
  * The verifier genuinely gates the repair: an unverified reflection is
    regenerated once, and if it still isn't trusted, the repair is done from
    the raw test output only (the bad diagnosis never reaches the model).

`check_code()` is a generator that yields event dicts so a UI can stream
progress.
"""
import ast
import json
import subprocess
import sys
import tempfile
import os
from typing import Iterator, List, Optional

from generators import generator_factory, model_factory
from generators.verifier import verify_reflection

TEST_TIMEOUT = 5
NO_DIAGNOSIS = ("No trusted diagnosis is available. Rely only on the failing "
                "test output above to find and fix the bug.")

_HARNESS = r'''
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


def split_tests(text: str) -> List[str]:
    """Each top-level statement of the tests text is one test."""
    try:
        tree = ast.parse(text)
        return [ast.get_source_segment(text, n) for n in tree.body]
    except SyntaxError:
        return [l.strip() for l in text.splitlines() if "assert" in l]


def run_tests(code: str, tests: List[str]) -> tuple:
    """Returns (all_passed, feedback_string, per_test_results)."""
    results = []
    with tempfile.TemporaryDirectory() as d:
        harness = os.path.join(d, "h.py")
        with open(harness, "w") as f:
            f.write(_HARNESS)
        for i, t in enumerate(tests):
            pf = os.path.join(d, f"p{i}.json")
            with open(pf, "w") as f:
                json.dump({"code": code, "test": t}, f)
            try:
                p = subprocess.run([sys.executable, "-I", harness, pf],
                                   capture_output=True, text=True,
                                   timeout=TEST_TIMEOUT, cwd=d)
                line = [l for l in p.stdout.splitlines() if l.startswith("@@RESULT@@")]
                r = json.loads(line[-1][10:]) if line else {
                    "ok": False, "output": (p.stderr or "no output")[-300:]}
            except subprocess.TimeoutExpired:
                r = {"ok": False, "output": "TIMEOUT"}
            results.append({"test": t, **r})
    fb = "Tested passed:"
    fb += "".join(f"\n{r['test']}" for r in results if r["ok"])
    fb += "\n\nTests failed:"
    fb += "".join(f"\n{r['test']} # output: {r['output']}" for r in results if not r["ok"])
    return all(r["ok"] for r in results), fb, results


def check_code(code: str, tests_text: str = "", description: str = "",
               model_name: str = "gemini-3.1-flash-lite",
               verifier_model_name: Optional[str] = None,
               max_iters: int = 3, use_verifier: bool = True,
               model=None, verifier_model=None) -> Iterator[dict]:
    gen = generator_factory("py")
    model = model or model_factory(model_name)
    if use_verifier:
        verifier_model = verifier_model or model_factory(verifier_model_name or model_name)

    spec = (description.strip() + "\n\n" if description.strip() else "") + code

    # 1. tests
    if tests_text.strip():
        tests, source = split_tests(tests_text), "user"
    else:
        tests, source = gen.internal_tests(spec, model, 5), "generated"
    if not tests:
        yield {"type": "error", "message": "No tests available — paste some asserts or add a description."}
        return
    yield {"type": "tests", "source": source, "tests": tests}

    # 2. run original
    cur = code
    passed, fb, res = run_tests(cur, tests)
    yield {"type": "run", "iter": 0, "code": cur, "passed": passed, "feedback": fb, "results": res}

    it = 1
    while not passed and it <= max_iters:
        # 3. reflect (+ verify, + one regeneration if untrusted)
        reflection = gen.self_reflection(cur, fb, model)
        trusted_reflection = reflection
        if use_verifier:
            v = verify_reflection(fb, reflection, verifier_model)
            yield {"type": "verdict", "iter": it, "attempt": 1, "reflection": reflection, **v}
            if not v["is_trusted"]:
                reflection = gen.self_reflection(cur, fb, model)
                v = verify_reflection(fb, reflection, verifier_model)
                yield {"type": "verdict", "iter": it, "attempt": 2, "reflection": reflection, **v}
            trusted_reflection = reflection if v["is_trusted"] else NO_DIAGNOSIS
        else:
            yield {"type": "reflection", "iter": it, "reflection": reflection}

        # 4. repair using only the trusted reflection
        cur = gen.func_impl(func_sig=spec, model=model, strategy="reflexion",
                            prev_func_impl=cur, feedback=fb,
                            self_reflection=trusted_reflection)
        passed, fb, res = run_tests(cur, tests)
        yield {"type": "run", "iter": it, "code": cur, "passed": passed, "feedback": fb, "results": res}
        it += 1

    yield {"type": "done", "passed": passed, "final_code": cur, "iterations": it - 1,
           "tests_source": source}
