"""
----------------------

Uses an LLM (Claude, via the Anthropic API) to turn EcoSight's numeric
outputs (forecast accuracy, cluster findings, carbon savings) into a
plain-English report a non-technical stakeholder could actually read.

This module is deliberately built to DEMONSTRATE prompt engineering as a
skill, not just "call an API":

1. `NAIVE_PROMPT` vs `build_engineered_prompt()` -- a vague one-line prompt
   is compared against a carefully constructed one, so the difference is
   visible and explainable, not just asserted.
2. The engineered prompt applies several standard techniques:
   - a system prompt establishing role, audience, and tone
   - explicit structure (asks for specific named sections)
   - explicit constraints (length, no jargon without explanation)
   - grounding: the actual numbers are injected into the prompt so the
     model reports on YOUR data rather than inventing plausible-sounding
     generic content (this matters a lot for factual reliability)
3. A `mock` fallback mode: if no API key is configured, this module still
   runs end-to-end using a clearly-labelled placeholder response, so the
   rest of the pipeline (and this demo) never breaks just because a key
   isn't set. Same fallback philosophy used for the data sources earlier.

To use this for real: get an API key from https://console.anthropic.com
and set it as an environment variable before running:
    export ANTHROPIC_API_KEY=your-key-here
"""

from __future__ import annotations

import os
import json
from pathlib import Path


NAIVE_PROMPT = "Write about the energy data."


def build_engineered_prompt(context: dict) -> tuple[str, str]:
    """
    Returns (system_prompt, user_prompt) for a well-engineered request.

    Techniques used:
      - Role/audience framing in the system prompt
      - Explicit output structure (named sections)
      - Explicit constraints (length, jargon handling)
      - Grounding: real numbers are injected, not left for the model to
        invent
    """
    system_prompt = (
        "You are an analyst preparing a briefing for IntelliGen's leadership "
        "team, who are business-savvy but not data scientists. Write in plain "
        "English. When you use a technical term (e.g. 'RMSE', 'SHAP', "
        "'genetic algorithm'), briefly explain it in the same sentence rather "
        "than assuming prior knowledge."
    )

    user_prompt = f"""Write a short briefing (250-350 words) on the EcoSight energy
monitoring system's results this period, based ONLY on the figures below --
do not invent or estimate any numbers not given here.

Structure it with these exact section headers:
1. What we measured
2. Key findings
3. Business impact
4. Caveats / limitations

Figures to report on:
- Forecasting: Random Forest MAE {context['rf_mae']:.1f} {context['unit']}, LSTM MAE {context['lstm_mae']:.1f} {context['unit']}
- Clustering: {context['n_clusters']} distinct daily demand patterns found across {context['n_days']} days analysed
- Scheduling optimisation: {context['carbon_saving_pct']:.1f}% carbon reduction achieved by the {context['scheduler_type']} scheduler versus a naive "start immediately" baseline
- Explainability: the top predictive feature was '{context['top_feature']}'
"""
    return system_prompt, user_prompt


def call_claude(system_prompt: str, user_prompt: str, model: str = "claude-sonnet-4-6",
                 max_tokens: int = 800, mock: bool | None = None) -> str:
    """
    Calls the Claude API. Falls back to a mock response if no API key is
    set (or mock=True is forced), so this module is always safe to run,
    even without credentials configured.

    Note: model names change over time -- check https://docs.claude.com
    for the current recommended model string before relying on this in
    production.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    use_mock = mock if mock is not None else (api_key is None)

    if use_mock:
        return (
            "[MOCK RESPONSE -- no ANTHROPIC_API_KEY set, so this is a placeholder "
            "showing where a real Claude-generated report would appear. Set the "
            "ANTHROPIC_API_KEY environment variable and re-run to get a real response.]\n\n"
            f"(System prompt used: {len(system_prompt)} chars, "
            f"user prompt used: {len(user_prompt)} chars)"
        )

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return response.content[0].text
    except Exception as e:
        return f"[ERROR calling Claude API: {e}]"


def generate_report(context: dict, mock: bool | None = None) -> dict:
    """
    Runs both the naive and engineered prompts on the same underlying data
    and returns both, so the difference can be shown side by side.
    """
    naive_response = call_claude(
        system_prompt="You are a helpful assistant.",
        user_prompt=NAIVE_PROMPT,
        mock=mock,
    )

    system_prompt, engineered_prompt = build_engineered_prompt(context)
    engineered_response = call_claude(
        system_prompt=system_prompt,
        user_prompt=engineered_prompt,
        mock=mock,
    )

    return {
        "naive_prompt": NAIVE_PROMPT,
        "naive_response": naive_response,
        "engineered_system_prompt": system_prompt,
        "engineered_user_prompt": engineered_prompt,
        "engineered_response": engineered_response,
    }


def save_comparison_markdown(result: dict, out_path: str):
    md = f"""# Prompt engineering comparison

## Naive prompt
**Prompt:** `{result['naive_prompt']}`

**Response:**
{result['naive_response']}

---

## Engineered prompt

**System prompt:**
```
{result['engineered_system_prompt']}
```

**User prompt:**
```
{result['engineered_user_prompt']}
```

**Response:**
{result['engineered_response']}

---

## Why the engineered version is better

The naive prompt gives the model no data to report on, no audience, no
structure, and no length limit -- so it can only produce generic, made-up
content about "energy data" in the abstract. It has no way to be accurate
about EcoSight specifically, because it was never told anything about
EcoSight.

The engineered prompt fixes this by:
- **Grounding**: the actual numbers are injected directly into the prompt,
  so the model reports on real results rather than inventing plausible
  ones -- critical for factual reliability in a business context.
- **Role and audience framing**: told who it's writing for (non-technical
  leadership), which shapes tone and how it explains jargon.
- **Explicit structure**: named sections make the output predictable and
  easy to drop into a report or slide.
- **Explicit constraints**: a word count keeps it usable in a real
  briefing rather than a rambling essay.
"""
    Path(out_path).write_text(md)
    print(f"Saved prompt comparison to {out_path}")


if __name__ == "__main__":
    # Example context -- in the full pipeline this would be populated from
    # the actual saved results of forecasting.py, clustering.py, and
    # scheduler.py / rl_scheduler.py rather than hardcoded.
    example_context = {
        "rf_mae": 393.4,
        "lstm_mae": 494.5,
        "unit": "MW",
        "n_clusters": 2,
        "n_days": 6027,
        "carbon_saving_pct": 14.8,
        "scheduler_type": "RL",
        "top_feature": "lag_168 (demand at the same hour, one week earlier)",
    }

    result = generate_report(example_context)

    print("=== NAIVE PROMPT RESPONSE ===")
    print(result["naive_response"])
    print("\n=== ENGINEERED PROMPT RESPONSE ===")
    print(result["engineered_response"])

    Path("docs").mkdir(exist_ok=True)
    save_comparison_markdown(result, "docs/prompt_comparison.md")
