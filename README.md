# Reflexion + Verifier 

## Structure

```
reflexion-verifier/
├── main.py                          # CLI entry point (simple | reflexion strategies)
├── reflexion.py                     # the core loop — verifier gate lives here
├── simple.py                        # baseline "no reflection" strategy (for comparison)
├── utils.py                         # jsonl I/O helpers
├── run_reflexion_with_verifier.sh   # ablation condition A
├── run_reflexion_baseline.sh        # ablation condition B (verifier off)
├── generators/
│   ├── __init__.py
│   ├── factory.py                   # model_factory() — registers Gemini + GPT4/35
│   ├── model.py                     # ModelBase, GeminiModel is in gemini_client.py
│   ├── gemini_client.py             # Gemini API wrapper (ModelBase interface)
│   ├── verifier.py                  # the Verifier Agent
│   ├── generator_types.py
│   ├── generator_utils.py           # generic_generate_self_reflection() etc — untouched
│   ├── py_generate.py               # Python-specific prompts
│   └── parse.py                     # code-block extraction from LLM output
├── executors/
│   ├── __init__.py
│   ├── factory.py                   # Python-only
│   ├── executor_types.py
│   ├── executor_utils.py
│   └── py_executor.py               # runs candidate code against tests
├── benchmarks/                      # benchmark file containing humaneval 
└── requirements.txt
```


## Setup

```bash
pip install -r requirements.txt
export GEMINI_API_KEY=your_key_here   # free at aistudio.google.com
```

## Running the ablation

```bash
chmod +x run_reflexion_with_verifier.sh run_reflexion_baseline.sh

./run_reflexion_with_verifier.sh    # condition A: verifier gates memory
./run_reflexion_baseline.sh          # condition B: plain Reflexion baseline
```

Both write to `root/<run_name>/*.jsonl`. Each row includes `is_solved`,
`reflections` (only the trusted ones, in the verifier condition),
`implementations`, `test_feedback`, and — in the verifier condition —
`verifier_log` (every reflection generated, its score, verdict, and
whether it was kept), which is what your report's cost/accuracy/ablation
metrics get computed from.

#
