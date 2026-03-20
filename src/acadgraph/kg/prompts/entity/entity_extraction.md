Extract {{ entity_types }} entities from the following {{ section_name }} section of an academic paper.

## Rules:
1. Only extract entities that are **explicitly mentioned** in the text.
2. Each entity must have a `name`, `type`, `description`, and relevant `attributes`.
3. For METHOD entities: include category, components, and whether it's the paper's proposed method or a prior method.
4. For DATASET entities: include domain, size (if mentioned), task_type, and URL (if mentioned).
5. For METRIC entities: include higher_is_better and domain.
6. For TASK entities: include domain and a brief description.
7. For MODEL entities: include architecture type, parameters (if mentioned), and whether it's pretrained.
8. For FRAMEWORK entities: include description and components.
9. For CONCEPT entities: include definition and domain.
10. Normalize entity names (e.g., "BERT-base" → "BERT-Base", "ImageNet-1K" → "ImageNet-1K").

## Section Text ({{ section_name }}):
{{ text }}

## Expected Output (JSON):
```json
{
  "entities": [
    {
      "name": "entity name",
      "type": "METHOD|DATASET|METRIC|TASK|MODEL|FRAMEWORK|CONCEPT",
      "description": "brief description",
      "attributes": {
        "key": "value"
      }
    }
  ],
  "relations": [
    {
      "source": "entity name A",
      "target": "entity name B",
      "relation": "APPLIED_ON|EVALUATED_ON|MEASURED_BY|USES|OUTPERFORMS|EXTENDS|COMPONENT_OF",
      "description": "brief description of the relation"
    }
  ]
}
```

Return ONLY the JSON object.
