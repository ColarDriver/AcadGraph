Extract {{ entity_types }} entities from the following {{ section_name }} section of an academic paper.

## Rules:
1. Only extract entities **explicitly mentioned** in the text.
2. Normalize names (e.g., "BERT-base" → "BERT-Base", "ImageNet-1K" → "ImageNet-1K").
3. Keep descriptions under 20 words.
4. Only include attributes with values actually stated in the text.
5. Limit to the **15 most important** entities per section.

## Section Text ({{ section_name }}):
{{ text }}

## Expected Output (JSON):
```json
{
  "entities": [
    {"name": "...", "type": "METHOD|DATASET|METRIC|TASK|MODEL|FRAMEWORK|CONCEPT", "description": "brief description"}
  ],
  "relations": [
    {"source": "entity A", "target": "entity B", "relation": "APPLIED_ON|EVALUATED_ON|MEASURED_BY|USES|OUTPERFORMS|EXTENDS|COMPONENT_OF"}
  ]
}
```

Return ONLY the JSON object. Keep output concise.
