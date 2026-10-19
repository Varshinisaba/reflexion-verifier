from generators.verifier import verify_reflection

from utils import enumerate_resume, make_printv, write_jsonl
from executors import executor_factory
from generators import generator_factory, model_factory

from typing import List

# Used for the retry when the verifier refuses to trust the reflection, so the
# bad diagnosis never reaches the coder (same behaviour as live_check.py).
NO_DIAGNOSIS = ("No trusted diagnosis is available. Rely only on the failing "
                "test output above to find and fix the bug.")


def run_reflexion(
    dataset: List[dict],
    model_name: str,
    language: str,
    max_iters: int,
    pass_at_k: int,
    log_path: str,
    verbose: bool,
    is_leetcode: bool = False,
    use_verifier: bool = True,
    verifier_model_name: str = "gemini-3.1-flash-lite",
) -> None:
    exe = executor_factory(language, is_leet=is_leetcode)
    gen = generator_factory(language)
    model = model_factory(model_name)
    verifier_model = model_factory(verifier_model_name) if use_verifier else None

    print_v = make_printv(verbose)

    # Stored in every row so runs can be audited / never silently mixed.
    config = {
        "model": model_name,
        "verifier_model": verifier_model_name if use_verifier else None,
        "use_verifier": use_verifier,
        "max_iters": max_iters,
        "pass_at_k": pass_at_k,
    }

    num_items = len(dataset)
    num_success = 0  # counts only tasks processed in THIS invocation
    num_done = 0
    for i, item in enumerate_resume(dataset, log_path):
        cur_pass = 0
        is_solved = False
        reflections = []      # trusted reflections only (verifier condition)
        verifier_log = []
        implementations = []
        test_feedback = []
        cur_func_impl = ""
        while cur_pass < pass_at_k and not is_solved:
            if is_leetcode:
                tests_i = item['visible_tests']
            else:
                tests_i = gen.internal_tests(item["prompt"], model, 1)

            # first attempt
            cur_func_impl = gen.func_impl(item["prompt"], model, "simple")
            implementations.append(cur_func_impl)
            assert isinstance(cur_func_impl, str)
            is_passing, feedback, _ = exe.execute(cur_func_impl, tests_i)
            test_feedback.append(feedback)

            # if solved, exit early
            if is_passing:
                is_passing = exe.evaluate(
                    item["entry_point"], cur_func_impl, item["test"], timeout=10)
                is_solved = is_passing
                break

            # use self-reflection to iteratively improve
            cur_iter = 1
            cur_feedback = feedback
            while cur_iter < max_iters:
                reflection = gen.self_reflection(
                    cur_func_impl, cur_feedback, model)

                # --- Verifier gate (REAL gate) -----------------------------------
                # An untrusted reflection is regenerated once; if it is still
                # untrusted, the retry is done from the raw test output only.
                # The untrusted text never reaches the coder, and is never kept.
                used_reflection = reflection
                if use_verifier:
                    result = verify_reflection(
                        evidence=cur_feedback,
                        reflection=reflection,
                        model=verifier_model,
                    )
                    verifier_log.append({
                        "iter": cur_iter, "attempt": 1,
                        "reflection": reflection, **result,
                    })
                    if not result["is_trusted"]:
                        reflection = gen.self_reflection(
                            cur_func_impl, cur_feedback, model)
                        result = verify_reflection(
                            evidence=cur_feedback,
                            reflection=reflection,
                            model=verifier_model,
                        )
                        verifier_log.append({
                            "iter": cur_iter, "attempt": 2,
                            "reflection": reflection, **result,
                        })
                    if result["is_trusted"]:
                        reflections.append(reflection)
                        used_reflection = reflection
                    else:
                        used_reflection = NO_DIAGNOSIS
                else:
                    reflections.append(reflection)
                # ------------------------------------------------------------------

                cur_func_impl = gen.func_impl(
                    func_sig=item["prompt"],
                    model=model,
                    strategy="reflexion",
                    prev_func_impl=cur_func_impl,
                    feedback=cur_feedback,
                    self_reflection=used_reflection,
                )
                implementations.append(cur_func_impl)
                assert isinstance(cur_func_impl, str)

                # check if all internal unit tests pass
                is_passing, cur_feedback, _ = exe.execute(
                    cur_func_impl, tests_i)
                test_feedback.append(cur_feedback)

                # if internal tests pass (or last iteration), check real tests
                if is_passing or cur_iter == max_iters - 1:
                    is_passing = exe.evaluate(
                        item["entry_point"], cur_func_impl, item["test"], timeout=10)
                    if is_passing:
                        is_solved = True
                    break

                cur_iter += 1
            cur_pass += 1

        num_success += int(is_solved)
        num_done += 1
        item["is_solved"] = is_solved
        item["reflections"] = reflections
        item["verifier_log"] = verifier_log
        item["implementations"] = implementations
        item["test_feedback"] = test_feedback
        item["solution"] = cur_func_impl
        item["config"] = config
        write_jsonl(log_path, [item], append=True)

        print_v(
            f'completed {i+1}/{num_items}: acc (this session) = '
            f'{round(num_success/num_done, 2)}')
