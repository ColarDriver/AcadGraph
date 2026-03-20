Analyze the following citations from an academic paper and classify each citation's intent.

## Citations:
{{ citations_json }}

## Intent Types:
1. CITES_FOR_PROBLEM: Problem/background motivation
2. CITES_AS_BASELINE: Experimental baseline comparison
3. CITES_FOR_FOUNDATION: Built on top of this work
4. CITES_AS_COMPARISON: Related work comparison
5. CITES_FOR_THEORY: Theoretical basis
6. EVOLVES_FROM: Explicitly improves upon

## Expected Output (JSON):
```json
{
  "classifications": [
    {
      "ref_key": "citation key",
      "intent": "INTENT_TYPE",
      "confidence": 0.0-1.0
    }
  ]
}
```

Return ONLY the JSON object.
