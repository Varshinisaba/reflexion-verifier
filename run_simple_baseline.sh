#!/bin/bash
# Condition A: NO reflection at all (single attempt). Needed to show what
# Reflexion itself adds before claiming anything about the verifier.
MODEL="${MODEL:-gemini-3.1-flash-lite}"
python main.py \
  --run_name "simple_no_reflection" \
  --root_dir "root" \
  --dataset_path ./benchmarks/humaneval-py.jsonl \
  --strategy "simple" \
  --language "py" \
  --model "$MODEL" \
  --pass_at_k "1" \
  --verbose
