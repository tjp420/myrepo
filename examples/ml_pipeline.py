"""ML pipeline example using scikit-learn.

Usage:
    python examples/ml_pipeline.py [train.csv]

If `train.csv` is not present, the script uses a tiny synthetic dataset.
"""
import sys
from pathlib import Path

def main():
    try:
        import pandas as pd
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import classification_report
    except Exception:
        print("Missing dependencies. Install with: pip install pandas scikit-learn")
        raise

    path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if path and path.exists():
        data = pd.read_csv(path)
    else:
        data = pd.DataFrame({
            "text": [
                "I love machine learning",
                "Databases are hard",
                "I enjoy reading about AI",
                "Database internals and indexes",
            ],
            "label": ["pos", "neg", "pos", "neg"],
        })

    X_train, X_test, y_train, y_test = train_test_split(
        data["text"], data["label"], test_size=0.25, random_state=42
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1, 2))),
        ("clf", LogisticRegression(max_iter=1000)),
    ])

    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    print(classification_report(y_test, preds))

if __name__ == "__main__":
    main()
