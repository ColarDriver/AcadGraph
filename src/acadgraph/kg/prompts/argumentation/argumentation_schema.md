Extract the argumentation structure from this academic paper.

## Paper Title: {{ title }}

## Abstract:
{{ abstract }}

## Introduction:
{{ introduction }}

## Method (summary):
{{ method_summary }}

## Conclusion:
{{ conclusion }}

## Task:
Extract the following elements:

### 1. PROBLEM
What specific research problem does this paper address?
- description: Clear statement of the problem
- scope: What domain/area does it apply to
- importance_signal: Why this problem matters

### 2. GAP
What gap in existing work does this paper identify?
- failure_mode: What specifically fails in current approaches
- constraint: Under what conditions does it fail
- prior_methods_failing: List of prior methods that have this failure

### 3. CORE_IDEA
What is the paper's key innovation/idea?
- mechanism: What the core mechanism is
- novelty_type: NEW_MECHANISM | NEW_FORMULATION | NEW_COMBINATION | NEW_APPLICATION | EFFICIENCY | THEORETICAL
- key_innovation: One-sentence description of what's new

### 4. CLAIMS (atomic, verifiable claims)
Break down the paper's main claims into atomic, testable statements.
Each claim should be:
- **Atomic**: Tests exactly one thing
- **Typed**: NOVELTY | PERFORMANCE | ROBUSTNESS | EFFICIENCY | THEORY | GENERALITY
- **Severity-rated**: 
  - P0: Critical — paper is invalid without evidence (e.g., "achieves SOTA")
  - P1: Important — significantly weakens paper if unverified
  - P2: Supporting — nice to have

## Expected Output (JSON):
```json
{
  "problem": {
    "description": "...",
    "scope": "...",
    "importance_signal": "..."
  },
  "gap": {
    "failure_mode": "...",
    "constraint": "...",
    "prior_methods_failing": ["method1", "method2"]
  },
  "core_idea": {
    "mechanism": "...",
    "novelty_type": "NEW_MECHANISM",
    "key_innovation": "..."
  },
  "claims": [
    {
      "text": "atomic claim statement",
      "type": "PERFORMANCE",
      "severity": "P0",
      "source_section": "abstract"
    }
  ]
}
```

## Rules:
1. Extract 3-8 atomic claims. Prefer quality over quantity.
2. Every P0 claim MUST be directly verifiable from the experiments section.
3. Performance claims with specific numbers should reference the metrics/datasets.
4. Novelty claims should be specific about what is new, not vague.
5. Do NOT invent claims — only extract what the paper actually states.

Return ONLY the JSON object.
