"""
Compare a with-verifier run against a without-verifier (baseline) run.

Usage:
    python compare_runs.py root/with_verifier/<file>.jsonl root/baseline_no_verifier/<file>.jsonl
"""
import json
import sys
from statistics import mean


def load(path):
    items = []
    with open(path) as f:
        for line in f:
            items.append(json.loads(line))
    return items


def summarize(items, label):
    n = len(items)
    solved = [i for i in items if i.get("is_solved")]
    pass_at_1 = len(solved) / n if n else 0.0

    # trials-to-success: number of implementations attempted, for solved tasks only
    #trials = [len(i["implementations"]) for i in solved if "implementations" in i]

    print(f"\n=== {label} ===")
    print(f"Tasks run:          {n}")
    print(f"Solved:              {len(solved)}")
    print(f"pass@1:              {pass_at_1:.2%}")
    #if trials:
    #    print(f"Avg trials-to-solve: {mean(trials):.2f}")

    verifier_logs = [v for i in items for v in i.get("verifier_log", [])]
    if verifier_logs:
        trusted = [v for v in verifier_logs if v.get("is_trusted")]
        scores = [v["score"] for v in verifier_logs if "score" in v]
        print(f"Reflections generated: {len(verifier_logs)}")
        print(f"Reflections trusted:    {len(trusted)} ({len(trusted)/len(verifier_logs):.0%})")
        print(f"Reflections discarded:  {len(verifier_logs) - len(trusted)}")
        if scores:
            print(f"Avg verifier score:     {mean(scores):.2f}")

    return {"n": n, "solved": len(solved), "pass_at_1": pass_at_1,
            #"avg_trials": mean(trials) if trials else None
            }


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python compare_runs.py <with_verifier.jsonl> <baseline.jsonl>")
        sys.exit(1)

    with_v = summarize(load(sys.argv[1]), "WITH verifier")
    without_v = summarize(load(sys.argv[2]), "WITHOUT verifier (baseline)")

    print(f"\n=== Delta ===")
    print(f"pass@1:      {with_v['pass_at_1']:+.2%} vs {without_v['pass_at_1']:.2%} "
          f"(diff: {with_v['pass_at_1'] - without_v['pass_at_1']:+.2%})")
    #if with_v["avg_trials"] and without_v["avg_trials"]:
    #    print(f"avg trials:  {with_v['avg_trials']:.2f} vs {without_v['avg_trials']:.2f} "
    #          f"(diff: {with_v['avg_trials'] - without_v['avg_trials']:+.2f})")
