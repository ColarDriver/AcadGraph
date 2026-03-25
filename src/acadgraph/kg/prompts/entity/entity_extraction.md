Extract {{ entity_types }} entities from the following {{ section_name }} section of an academic paper.

## Rules:
1. Only extract entities **explicitly mentioned** in the text.
2. **CRITICAL — Name Normalization**:
   - Use the **canonical, short, widely-recognized name** for each entity.
   - Method/Model names: use the **official name or acronym** (e.g., "Adam", "ResNet", "BERT", "GAN", "PPO", "SGD").
   - Do NOT use full descriptions as names. Maximum **5 words** per entity name.
   - If the paper proposes a new method, use the **proposed acronym/name** from the paper.
   - Normalize casing: "reinforcement learning" → "Reinforcement Learning".
   - Merge abbreviations with full names: always prefer "GNN" over "Graph Neural Network" if both appear.
   - Dataset names: use official names (e.g., "ImageNet", "CIFAR-10", "GLUE").
   - Task names: use standard terms (e.g., "Image Classification", "Object Detection", "Machine Translation").
3. Keep descriptions under 20 words.
4. Only include attributes with values actually stated in the text.
5. Limit to the **15 most important** entities per section.

### Bad ❌ vs Good ✅ Entity Names:
- ❌ "Novel mapping of features from the image domain to the 3D robot coordinate frame" → ✅ "Feature Mapping"
- ❌ "Multi-Objective Robust Bilevel Two-timescale optimization algorithm" → ✅ "MORBT" (or the paper's acronym)
- ❌ "Stochastic Gradient Descent optimization method" → ✅ "SGD"
- ❌ "convolutional neural network for image recognition" → ✅ "CNN"

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
