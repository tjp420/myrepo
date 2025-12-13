"""Dev shim for fact checking components referenced in api_server."""

from typing import Any, Dict, List, Optional


class Claim:
    def __init__(self, text: str = "", metadata: Optional[Dict[str, Any]] = None):
        self.text = text
        self.metadata = metadata or {}


class Evidence:
    def __init__(
        self,
        src: str = "",
        description: str = "",
        type: str = "fact",
        weight: int = 1,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.src = src
        self.description = description
        self.type = type
        self.weight = weight
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "src": self.src,
            "description": self.description,
            "type": self.type,
            "weight": self.weight,
            "metadata": self.metadata,
        }


class EvidenceType:
    FACT = "fact"
    OPINION = "opinion"


class EvidenceWeight:
    HIGH = 3
    MEDIUM = 2
    LOW = 1


class FactChecker:
    def __init__(self) -> None:
        self.claims: List[Claim] = []
        self.evidence: List[Evidence] = []

    def register_claim(self, claim: Claim) -> None:
        self.claims.append(claim)

    def add_evidence(self, evidence: Evidence) -> None:
        self.evidence.append(evidence)

    def analyze_claim(self, claim: Claim) -> Dict[str, Any]:
        # very small heuristic stub for dev-mode static checks
        return {
            "claim": claim.text,
            "verified": False,
            "evidence_count": len(self.evidence),
        }

    def verify(self, claim: Claim) -> bool:
        # default to False in dev shims
        return False


__all__ = ["FactChecker", "Claim", "Evidence", "EvidenceType", "EvidenceWeight"]
"""Stub for fact_checker used by api_server."""


def check_fact(statement: str) -> bool:
    return True
