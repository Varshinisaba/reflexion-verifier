"""
Verifier Agent — scores a self-written reflection against the real failure
evidence (unit test feedback) before it's trusted for reuse.

Matches ModelBase.generate(prompt, ...) -> str, the interface every model
in this project (GeminiModel, GPT4, etc.) implements — so this works with
whichever model you pass in, unchanged.
"""

import json
import re

VERIFIER_PROMPT = """You are a Verifier Agent auditing a code-repair agent's self-written reflection.

You will be given:
1. FAILURE EVIDENCE — the real unit test results from running the agent's code
2. REFLECTION — the agent's own diagnosis of what went wrong

Judge ONLY whether the reflection's claims are factually consistent with the
evidence. Do not judge whether its proposed fix would work — only whether
its diagnosis is grounded in what actually failed.

Respond with ONLY strict JSON, no markdown fences, no other text:
{{"score": <float 0.0-1.0>, "verdict": "SUPPORTED" | "PARTIALLY_SUPPORTED" | "UNSUPPORTED", "justification": "<one sentence>"}}

FAILURE EVIDENCE:
{evidence}

REFLECTION:
{reflection}
"""


def verify_reflection(evidence: str, reflection: str, model, trust_threshold: float = 0.7) -> dict:
    """
    `model` is any ModelBase instance (GeminiModel, GPT4, ...) — must
    implement .generate(prompt, max_tokens=..., temperature=...) -> str.

    Returns: {"score": float, "verdict": str, "justification": str, "is_trusted": bool}
    """
    prompt = VERIFIER_PROMPT.format(
        evidence=str(evidence)[:3000],
        reflection=str(reflection)[:1500],
    )
    try:
        raw = model.generate(prompt, max_tokens=2048, temperature=0.0)
    except Exception as e:  # API failure must not kill a long run; fail closed
        return {"score": 0.0, "verdict": "UNSUPPORTED",
                "justification": f"Verifier call failed: {e}", "is_trusted": False}
    if isinstance(raw, list):
        raw = raw[0]

    parsed = _parse_json_safely(raw)
    if parsed is None:
        # Fail closed: an unparsable verdict is never trusted.
        return {
            "score": 0.0,
            "verdict": "UNSUPPORTED",
            "justification": f"Verifier output could not be parsed: {raw[:200]!r}",
            "is_trusted": False,
        }

    if not isinstance(parsed, dict):
        return {"score": 0.0, "verdict": "UNSUPPORTED",
                "justification": f"Verifier returned non-object JSON: {str(parsed)[:200]!r}",
                "is_trusted": False}
    try:
        score = max(0.0, min(1.0, float(parsed.get("score", 0.0))))
    except (TypeError, ValueError):
        score = 0.0  # fail closed on a malformed score
    return {
        "score": score,
        "verdict": parsed.get("verdict", "UNSUPPORTED"),
        "justification": parsed.get("justification", ""),
        "is_trusted": score >= trust_threshold,
    }


def _parse_json_safely(text: str):
    text = re.sub(r"```json|```", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                return None
        return None
