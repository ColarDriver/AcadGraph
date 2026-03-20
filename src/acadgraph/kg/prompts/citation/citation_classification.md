Analyze the following citation context from the "{{ section }}" section of an academic paper and classify the citation intent.

## Citation Context:
"{{ citation_context }}"

## Cited Paper: {{ cited_title }}

## Intent Types:
1. **CITES_FOR_PROBLEM**: The citation is used to describe or motivate the research problem/background.
2. **CITES_AS_BASELINE**: The cited work is used as an experimental baseline for comparison.
3. **CITES_FOR_FOUNDATION**: The current work is built on top of / extends the cited work.
4. **CITES_AS_COMPARISON**: Horizontal comparison in related work discussion.
5. **CITES_FOR_THEORY**: The citation provides theoretical basis or proof technique.
6. **EVOLVES_FROM**: The current method explicitly evolves from or improves upon the cited work.

## Classification Rules:
- If in Introduction with problem/motivation context → likely CITES_FOR_PROBLEM
- If in Experiments with performance comparison → likely CITES_AS_BASELINE
- If in Method with "we build upon" / "extending" language → likely CITES_FOR_FOUNDATION or EVOLVES_FROM
- If in Related Work with comparison language → likely CITES_AS_COMPARISON
- If citing a theorem, proof, or mathematical framework → likely CITES_FOR_THEORY
- EVOLVES_FROM requires explicit language like "improves upon", "extends", "builds on"

## Expected Output (JSON):
```json
{
  "intent": "CITES_FOR_PROBLEM|CITES_AS_BASELINE|CITES_FOR_FOUNDATION|CITES_AS_COMPARISON|CITES_FOR_THEORY|EVOLVES_FROM",
  "confidence": 0.0-1.0,
  "reasoning": "brief explanation"
}
```

Return ONLY the JSON object.
