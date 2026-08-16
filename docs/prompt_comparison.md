# Prompt Engineering Comparison

## Naive Prompt

**Prompt:**

`Write about the energy data.`

### Response

[ERROR calling OpenAI API: Error code: 429 - {'error': {'message': 'You exceeded your current quota, please check your plan and billing details. For more information on this error, read the docs: https://platform.openai.com/docs/guides/error-codes/api-errors.', 'type': 'insufficient_quota', 'param': None, 'code': 'insufficient_quota'}}]

---

## Engineered Prompt

### Instructions

```text
You are a senior energy and data analyst preparing a briefing for IntelliGen's leadership team. The audience is business-savvy but does not have specialist knowledge of machine learning or data science. 

Write in clear, professional, plain English. When you use a technical term such as MAE, Random Forest, LSTM, clustering, reinforcement learning, or SHAP, briefly explain what it means in the same sentence. 

Your job is to interpret the supplied EcoSight results accurately, not to invent additional findings.

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
