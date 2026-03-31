from dataclasses import dataclass, field
from typing import Set, FrozenSet, Tuple

@dataclass
class SemanticMapping:
    """
    Dataset-specific semantic class co-occurrence mapping.
    
    Contributors: add a new file in semantic_maps/ defining a SemanticMapping
    instance and register it in semantic_maps/__init__.py under your dataset name.
    
    co_occurrence_pairs: unordered frozensets of class name pairs that are known
    to genuinely co-occur in the same anatomical region in this dataset.
    """
    dataset_name: str
    co_occurrence_pairs: Set[FrozenSet[str]] = field(default_factory=set)

    @classmethod
    def from_pairs(cls, dataset_name: str, pairs: list[Tuple[str, str]]) -> "SemanticMapping":
        return cls(
            dataset_name=dataset_name,
            co_occurrence_pairs={frozenset(pair) for pair in pairs}
        )

    def are_semantically_related(self, class_a: str, class_b: str) -> bool:
        return frozenset([class_a, class_b]) in self.co_occurrence_pairs

# ===================Add Dataset-specific semantic maps here======================================================

VINDRCXR_SEMANTIC_MAP = SemanticMapping.from_pairs(
    dataset_name="vindrcxr",
    pairs=[
        ("ILD", "Pulmonary Fibrosis"),           # subtype relationship
        ("Cardiomegaly", "Aortic Enlargement"),   # same mediastinal region
        ("Pleural Effusion", "Atelectasis"),       # lower zone co-occurrence
        ("Consolidation", "Pneumonia"),            # causal relationship
        ("Cardiomegaly", "Pulmonary Edema"),       # heart failure co-occurrence
    ]
)

# ===================End of Dataset-specific semantic maps here====================================================



def get_semantic_map(dataset_name: str) -> SemanticMapping | None:
    return SEMANTIC_MAPS.get(dataset_name.lower())


# ===============Add new semantic maps to the dictionary here=====================================================
SEMANTIC_MAPS = {
    "vindrcxr": VINDRCXR_SEMANTIC_MAP,
}