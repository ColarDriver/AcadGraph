Classify the rhetorical role of each paragraph in the following academic paper text.

## Rhetorical Roles:
- MOTIVATION: Why the problem matters, real-world importance
- BACKGROUND: Factual background, definitions, prior work (neutral)
- CONTRIBUTION: Paper's contributions, novelty claims, "we propose"
- METHOD_DESC: Proposed method, algorithm, architecture
- RESULT: Experimental results, numbers, tables, performance
- COMPARISON: Compare with baselines, relative performance
- LIMITATION: Limitations, failure cases, caveats
- FUTURE: Future directions, open problems, next steps

## Paper Text:
{{ text }}

## Instructions:
- Paragraphs are separated by double newlines.
- Return ONE role per paragraph, in order.
- Use the COMPACT format below (one role per paragraph, no extra fields).

## Expected Output (JSON):
```json
{"roles": ["MOTIVATION", "BACKGROUND", "METHOD_DESC", "RESULT"]}
```

Return ONLY the JSON object. The "roles" array length must equal the number of paragraphs.
