Assess the evidence support for each claim in this academic paper.

## Paper Title: {{ title }}

## Claims to verify:
{{ claims_json }}

## Experiments Section:
{{ experiments_text }}

## Tables:
{{ tables_text }}

## Task:
For each claim, find all relevant evidence from the experiments and assess support strength.

### Evidence Types:
- EXPERIMENT: Standard experimental result
- ABLATION: Ablation study result
- THEOREM: Theoretical proof or guarantee
- CASE_STUDY: Qualitative case study / example
- USER_STUDY: Human evaluation study
- ANALYSIS: Analysis experiment (e.g., robustness, convergence)

### Support Strength:
- **FULL**: The evidence directly and completely supports the claim with clear numerical/logical proof.
- **PARTIAL**: The evidence provides some support but not complete (e.g., only on some datasets, missing edge cases).
- **REFUTED**: The evidence actually contradicts or weakens the claim.
- **UNVERIFIABLE**: No relevant evidence found to verify this claim.

## Expected Output (JSON):
```json
{
  "evidence_links": [
    {
      "claim_index": 0,
      "evidences": [
        {
          "evidence_type": "EXPERIMENT",
          "result_summary": "description of the evidence",
          "datasets": ["dataset1", "dataset2"],
          "metrics": ["metric1", "metric2"],
          "tables": ["Table 1"],
          "figures": ["Figure 3"],
          "numeric_results": {"BLEU": 32.5, "ROUGE-L": 45.2},
          "support_strength": "FULL|PARTIAL|REFUTED|UNVERIFIABLE",
          "explanation": "why this strength level"
        }
      ]
    }
  ],
  "baselines": [
    {
      "method_name": "baseline method name",
      "paper_ref": "citation key or paper title",
      "performance": {"metric": "value"}
    }
  ],
  "numeric_consistency_issues": [
    {
      "claim_text": "claim that mentions a number",
      "claimed_value": "value in text",
      "table_value": "value in table",
      "consistent": true/false
    }
  ]
}
```

## Rules:
1. Be strict: a P0 claim with only partial evidence is a RED FLAG.
2. Check numeric consistency: if the abstract says "improves by 5%" but the table shows 3%, flag it.
3. Look for missing baselines: if a SOTA claim doesn't compare with the actual current SOTA, note it.
4. Ablation studies that don't test the core component → the mechanism claim is PARTIAL at best.
5. If a claim is about "all datasets" but evidence only covers 3 out of 5, that's PARTIAL.

Return ONLY the JSON object.
