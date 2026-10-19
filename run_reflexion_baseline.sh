#!/bin/bash
# Condition B: plain Reflexion, verifier OFF. Same settings as the verifier run.
MODEL="${MODEL:-gemini-3.1-flash-lite}"
python main.py \
  --run_name "baseline_reflexion_final" \
  --root_dir "root" \
  --dataset_path ./benchmarks/humaneval-py.jsonl \
  --strategy "reflexion" \
  --language "py" \
  --model "$MODEL" \
  --pass_at_k "1" \
  --max_iters "4" \
  --use_verifier "false" \
  --verbose
