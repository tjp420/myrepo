"""NLP normalization example.

Usage:
    python examples/nlp_normalize.py [entities.csv]

If `entities.csv` is not present, the script runs a small built-in example.
"""
import sys
from pathlib import Path

def main():
    try:
        import pandas as pd
        import spacy
    except Exception as exc:
        print("Missing dependencies. Install with: pip install pandas spacy")
        raise

    nlp = spacy.load("en_core_web_sm")

    path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if path and path.exists():
        df = pd.read_csv(path)
    else:
        df = pd.DataFrame([
            {"text": "Alice studies Machine Learning and Databases", "entity": "Alice", "concept": "MachineLearning"},
            {"text": "Bob works on Databases and Systems", "entity": "Bob", "concept": "Databases"},
        ])

    def normalize_text(text: str) -> str:
        doc = nlp(str(text).lower())
        tokens = [token.lemma_ for token in doc if not token.is_punct and not token.is_space]
        return " ".join(tokens)

    df["text_norm"] = df["text"].apply(normalize_text)
    print(df["text_norm"])

if __name__ == "__main__":
    main()
