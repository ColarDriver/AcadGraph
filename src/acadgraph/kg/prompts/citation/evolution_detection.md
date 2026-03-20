Given the following method descriptions from papers ordered by year, identify evolution relationships.

## Methods (chronological):
{{ methods_json }}

## Task:
For each pair of methods where the later one explicitly builds upon or evolves from the earlier one, identify the evolution link.

## Expected Output (JSON):
```json
{
  "evolution_links": [
    {
      "from_method": "earlier method name",
      "to_method": "later method name",
      "delta_description": "what changed/improved",
      "confidence": 0.0-1.0
    }
  ]
}
```

Return ONLY the JSON object.
