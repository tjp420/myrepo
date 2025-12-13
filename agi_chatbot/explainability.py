"""Dev stub for explainability engine integration."""


def get_explainability_engine():
    class _E:
        def explain(self, *a, **k):
            return {"explanation": ""}

        def get_explanations(self, *a, **k):
            return []

    return _E()


__all__ = ["get_explainability_engine"]
