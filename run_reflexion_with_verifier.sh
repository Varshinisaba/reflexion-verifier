#!/bin/bash
# Condition C: Reflexion WITH the Verifier gate (untrusted reflections never reach the coder).
# Same model / dataset / max_iters as the other two conditions -> directly comparable.
# Requires GEMINI_API_KEY.
MODEL="${MODEL:-gemini-3.1-flash-lite}"
python main.py \
  --run_name "with_verifier_final" \
  --root_dir "root" \
  --dataset_path ./benchmarks/humaneval-py.jsonl \
  --strategy "reflexion" \
  --language "py" \
  --model "$MODEL" \
  --pass_at_k "1" \
  --max_iters "4" \
  --use_verifier "true" \
  --verifier_model "$MODEL" \
  --verbose
