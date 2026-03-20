Classify the rhetorical role of each paragraph in the following academic paper text.

## Rhetorical Roles:
1. **MOTIVATION**: Paragraphs that describe why the problem matters, real-world importance, applications.
2. **BACKGROUND**: Paragraphs that provide factual background, definitions, prior work description (neutral).
3. **CONTRIBUTION**: Paragraphs that state the paper's contributions, novelty claims, "we propose" statements.
4. **METHOD_DESC**: Paragraphs that describe the proposed method, algorithm, architecture, or technique.
5. **RESULT**: Paragraphs that present experimental results, numbers, tables, performance outcomes.
6. **COMPARISON**: Paragraphs that compare with baselines, discuss relative performance, pros/cons vs prior work.
7. **LIMITATION**: Paragraphs that discuss limitations, failure cases, caveats, or shortcomings.
8. **FUTURE**: Paragraphs that discuss future directions, open problems, or next steps.

## Paper Text:
{{ text }}

## Instructions:
- Assign exactly ONE role to each paragraph.
- A paragraph is separated by double newlines.
- Focus on the PRIMARY purpose of each paragraph.
- If a paragraph mixes roles, choose the dominant one.

## Expected Output (JSON):
```json
{
  "paragraphs": [
    {
      "index": 0,
      "first_sentence": "first 50 chars of the paragraph...",
      "role": "MOTIVATION|BACKGROUND|CONTRIBUTION|METHOD_DESC|RESULT|COMPARISON|LIMITATION|FUTURE",
      "confidence": 0.0-1.0
    }
  ]
}
```

Return ONLY the JSON object.
