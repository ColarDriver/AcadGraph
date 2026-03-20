You are given two entity descriptions from academic papers. Determine if they refer to the same entity.

Entity A:
- Name: {{ name_a }}
- Type: {{ type_a }}
- Description: {{ desc_a }}

Entity B:
- Name: {{ name_b }}
- Type: {{ type_b }}
- Description: {{ desc_b }}

Return JSON:
```json
{
  "is_same": true/false,
  "merged_name": "canonical name if same entity",
  "confidence": 0.0-1.0
}
```
