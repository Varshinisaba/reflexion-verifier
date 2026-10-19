import os
import argparse
from simple import run_simple
from reflexion import run_reflexion
from utils import read_jsonl, read_jsonl_gz


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, help="The name of the run")
    parser.add_argument("--root_dir", type=str,
                        help="The root logging directory", default="root")
    parser.add_argument("--dataset_path", type=str,
                        help="The path to the benchmark dataset", default="root")
    parser.add_argument("--strategy", type=str,
                        help="Strategy: `simple`, `reflexion`")
    parser.add_argument("--language", type=str, help="`py` (only supported language in this trimmed project)", default="py")
    parser.add_argument(
        "--model", type=str, help="Model for the coder/reflector, e.g. `gemini-3.6-flash` or `gpt-4`")
    parser.add_argument("--pass_at_k", type=int,
                        help="Pass@k metric", default=1)
    parser.add_argument("--max_iters", type=int,
                        help="The maximum number of self-improvement iterations", default=10)

    parser.add_argument("--is_leetcode", action='store_true',
                        help="To run the leetcode benchmark")  # Temporary

    parser.add_argument("--verbose", action='store_true',
                        help="To print live logs")

    # --- Verifier Agent controls ---
    parser.add_argument("--use_verifier", type=lambda x: x.lower() == "true",
                        default=True,
                        help="Gate self-reflections through a Verifier Agent "
                             "before they're reused (true/false, default true). "
                             "Set false to reproduce plain baseline Reflexion.")
    parser.add_argument("--verifier_model", type=str, default="gemini-3.1-flash-lite",
                        help="Model used for the Verifier Agent")

    args = parser.parse_args()
    return args


def strategy_factory(strategy: str):
    def kwargs_wrapper_gen(func, delete_keys=[]):
        def kwargs_wrapper(**kwargs):
            for key in delete_keys:
                del kwargs[key]
            return func(**kwargs)
        return kwargs_wrapper

    if strategy == "simple":
        return kwargs_wrapper_gen(run_simple, delete_keys=["max_iters", "use_verifier", "verifier_model_name"])
    elif strategy == "reflexion":
        return kwargs_wrapper_gen(run_reflexion)
    else:
        raise ValueError(f"Strategy `{strategy}` is not supported in this "
                          f"trimmed project (only `simple`, `reflexion`)")


def main(args):
    if not os.path.exists(args.root_dir):
        os.makedirs(args.root_dir)

    dataset_name = os.path.basename(args.dataset_path).replace("jsonl", "")

    log_dir = os.path.join(args.root_dir, args.run_name)
    log_path = os.path.join(
        log_dir,
        f"{dataset_name}_{args.strategy}_{args.max_iters}_{args.model}"
        f"_verifier-{args.use_verifier}_pass_at_k_{args.pass_at_k}_{args.language}.jsonl")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    run_strategy = strategy_factory(args.strategy)

    if args.verbose:
        print(f"""
Starting run with the following parameters:
strategy: {args.strategy}
model: {args.model}
use_verifier: {args.use_verifier}
pass@k: {args.pass_at_k}
""")
    else:
        print(f"Logs will be saved in `{log_dir}`")

    print(f'Loading the dataset...')
    if args.dataset_path.endswith(".jsonl"):
        dataset = read_jsonl(args.dataset_path)
    elif args.dataset_path.endswith(".jsonl.gz"):
        dataset = read_jsonl_gz(args.dataset_path)
    else:
        raise ValueError(
            f"Dataset path `{args.dataset_path}` is not supported")

    print(f"Loaded {len(dataset)} examples")

    run_strategy(
        dataset=dataset,
        model_name=args.model,
        language=args.language,
        max_iters=args.max_iters,
        pass_at_k=args.pass_at_k,
        log_path=log_path,
        verbose=args.verbose,
        is_leetcode=args.is_leetcode,
        use_verifier=args.use_verifier,
        verifier_model_name=args.verifier_model,
    )

    print(f"Done! Check out the logs in `{log_path}`")


if __name__ == "__main__":
    args = get_args()
    main(args)
