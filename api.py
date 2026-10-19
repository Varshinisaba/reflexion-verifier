"""
FastAPI backend for the Reflexion + Verifier dashboard.

Wraps reflexion.py / simple.py directly (no subprocess) so runs share this
process's Python environment and the Gemini client's rate-limit throttling.

Run:
    pip install fastapi uvicorn
    uvicorn api:app --reload --port 8000
"""

import os
import json
import threading
import traceback
from datetime import datetime
from statistics import mean

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from utils import read_jsonl
from reflexion import run_reflexion
from simple import run_simple
from live_check import check_code

app = FastAPI(title="Reflexion + Verifier Dashboard API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)

ROOT_DIR = "root"
os.makedirs(ROOT_DIR, exist_ok=True)

# In-memory run tracking: run_name -> {"status", "started_at", "error", "log_path"}
_RUNS: dict = {}
_RUNS_LOCK = threading.Lock()


class RunRequest(BaseModel):
    run_name: str
    dataset_path: str = "./benchmarks/humaneval-py.jsonl"
    strategy: str = "reflexion"       # "reflexion" | "simple"
    model: str = "gemini-3.5-flash-lite"
    max_iters: int = 4
    pass_at_k: int = 1
    use_verifier: bool = True
    verifier_model: str = "gemini-3.5-flash-lite"
    limit: int | None = None          # cap number of tasks, for cheap test runs


def _execute_run(req: RunRequest):
    log_path = os.path.join(ROOT_DIR, req.run_name,
                             f"{req.run_name}.jsonl")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    with _RUNS_LOCK:
        _RUNS[req.run_name] = {
            "status": "running", "started_at": datetime.utcnow().isoformat(),
            "error": None, "log_path": log_path,
        }

    try:
        dataset = read_jsonl(req.dataset_path)
        if req.limit:
            dataset = dataset[: req.limit]

        if req.strategy == "simple":
            run_simple(
                dataset=dataset, model_name=req.model, language="py",
                pass_at_k=req.pass_at_k, log_path=log_path, verbose=False,
                is_leetcode=False,
            )
        else:
            run_reflexion(
                dataset=dataset, model_name=req.model, language="py",
                max_iters=req.max_iters, pass_at_k=req.pass_at_k,
                log_path=log_path, verbose=False, is_leetcode=False,
                use_verifier=req.use_verifier,
                verifier_model_name=req.verifier_model,
            )
        with _RUNS_LOCK:
            _RUNS[req.run_name]["status"] = "done"
    except Exception:
        with _RUNS_LOCK:
            _RUNS[req.run_name]["status"] = "error"
            _RUNS[req.run_name]["error"] = traceback.format_exc()


@app.post("/api/run")
def start_run(req: RunRequest):
    with _RUNS_LOCK:
        if req.run_name in _RUNS and _RUNS[req.run_name]["status"] == "running":
            raise HTTPException(400, f"Run '{req.run_name}' is already in progress.")

    thread = threading.Thread(target=_execute_run, args=(req,), daemon=True)
    thread.start()
    return {"status": "started", "run_name": req.run_name}


@app.get("/api/status/{run_name}")
def get_status(run_name: str):
    with _RUNS_LOCK:
        info = _RUNS.get(run_name)
    if info is None:
        raise HTTPException(404, "Unknown run_name (not started this session).")
    return info


@app.get("/api/runs")
def list_runs():
    """Every run folder found under root/, regardless of whether it was
    started this session (so past runs from CLI usage show up too)."""
    if not os.path.exists(ROOT_DIR):
        return []
    names = []
    for entry in sorted(os.listdir(ROOT_DIR)):
        run_dir = os.path.join(ROOT_DIR, entry)
        if os.path.isdir(run_dir):
            jsonl_files = [f for f in os.listdir(run_dir) if f.endswith(".jsonl")]
            if jsonl_files:
                names.append(entry)
    return names


def _load_run_results(run_name: str) -> list[dict]:
    run_dir = os.path.join(ROOT_DIR, run_name)
    if not os.path.isdir(run_dir):
        raise HTTPException(404, f"No run directory for '{run_name}'.")
    jsonl_files = [f for f in os.listdir(run_dir) if f.endswith(".jsonl")]
    if not jsonl_files:
        raise HTTPException(404, f"No results file yet for '{run_name}'.")
    results = []
    for f in jsonl_files:
        with open(os.path.join(run_dir, f)) as fh:
            for line in fh:
                if line.strip():
                    results.append(json.loads(line))
    return results


@app.get("/api/results/{run_name}")
def get_results(run_name: str):
    return _load_run_results(run_name)


def _summarize(items: list[dict]) -> dict:
    n = len(items)
    solved = [i for i in items if i.get("is_solved")]
    #trials = [len(i["implementations"]) for i in solved if "implementations" in i]
    verifier_logs = [v for i in items for v in i.get("verifier_log", [])]
    trusted = [v for v in verifier_logs if v.get("is_trusted")]

    return {
        "n_tasks": n,
        "n_solved": len(solved),
        "pass_at_1": (len(solved) / n) if n else 0.0,
        #"avg_trials_to_solve": mean(trials) if trials else None,
        "n_reflections_generated": len(verifier_logs),
        "n_reflections_trusted": len(trusted),
        "reflection_trust_rate": (len(trusted) / len(verifier_logs)) if verifier_logs else None,
    }


@app.get("/api/compare")
def compare(run_a: str, run_b: str):
    items_a = _load_run_results(run_a)
    items_b = _load_run_results(run_b)

    by_task_a = {i["name"]: i.get("is_solved", False) for i in items_a}
    by_task_b = {i["name"]: i.get("is_solved", False) for i in items_b}
    common_tasks = sorted(set(by_task_a) & set(by_task_b))

    flips = []
    for task in common_tasks:
        a, b = by_task_a[task], by_task_b[task]
        if a != b:
            flips.append({"task": task, "run_a_solved": a, "run_b_solved": b})

    return {
        "run_a": {"name": run_a, "summary": _summarize(items_a)},
        "run_b": {"name": run_b, "summary": _summarize(items_b)},
        "common_tasks": len(common_tasks),
        "flips": flips,
    }


class CheckRequest(BaseModel):
    code: str
    tests: str = ""
    description: str = ""
    model: str = "gemini-3.1-flash-lite"
    verifier_model: str = "gemini-3.1-flash-lite"
    max_iters: int = 3
    use_verifier: bool = True


@app.post("/api/check")
def check(req: CheckRequest):
    """Paste code -> stream Reflexion+Verifier events (Server-Sent Events)."""
    if not req.code.strip():
        raise HTTPException(400, "code is empty")

    def stream():
        try:
            for ev in check_code(req.code, req.tests, req.description,
                                 req.model, req.verifier_model,
                                 req.max_iters, req.use_verifier):
                yield f"data: {json.dumps(ev)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.get("/api/health")
def health():
    return {"status": "ok"}
