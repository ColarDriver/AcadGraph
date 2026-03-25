Extract all cited works from the following academic paper sections. For each citation, identify:
1. The **title** of the cited work (as specifically as possible)
2. The **section** where it is cited
3. The **intent** of the citation

## Paper Sections:
{{ sections_text }}

## Intent Types:
1. CITES_FOR_PROBLEM: Problem/background motivation
2. CITES_AS_BASELINE: Experimental baseline comparison
3. CITES_FOR_FOUNDATION: Built on top of this work
4. CITES_AS_COMPARISON: Related work comparison
5. CITES_FOR_THEORY: Theoretical basis
6. EVOLVES_FROM: Explicitly improves upon this work

## Rules:
- Only extract works that are **explicitly cited** in the text
- Use the **exact title** if mentioned, otherwise use "Author et al. (Year)" format
- Do NOT invent citations. Only extract what's in the text.
- Maximum **30 citations** per paper
- Focus on the most important citations (skip generic "see also" references)

## Expected Output (JSON):
```json
{
  "citations": [
    {
      "cited_title": "Title of the cited paper",
      "cited_authors": "First Author et al.",
      "cited_year": 2023,
      "section": "introduction|related_work|method|experiments",
      "intent": "INTENT_TYPE",
      "context": "The sentence where the citation appears (max 200 chars)"
    }
  ]
}
```

Return ONLY the JSON object.
