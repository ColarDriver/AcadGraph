"""Knowledge extraction sub-package.

Contains the three extraction modules:
- entities: Semantic entity extraction (METHOD, DATASET, METRIC, TASK, etc.)
- evolution: Citation relation classification & technology evolution chains
- argumentation: Argumentation chain extraction (3-Pass Pipeline)
"""

from acadgraph.kg.extract.entities import EntityExtractor
from acadgraph.kg.extract.evolution import CitationEvolutionBuilder
from acadgraph.kg.extract.argumentation import ArgumentationExtractor

__all__ = [
    "EntityExtractor",
    "CitationEvolutionBuilder",
    "ArgumentationExtractor",
]
