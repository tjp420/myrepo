# Examples

This folder contains small runnable example scripts demonstrating common tasks.

Setup
1. Create and activate a virtual environment (recommended).
2. Install the example dependencies from the project root:

```powershell
pip install -r requirements.txt
# If using the NLP example, also install the spaCy model:
python -m spacy download en_core_web_sm
```

Examples
- `examples/nlp_normalize.py` — normalize text using spaCy and build a tiny knowledge graph.
- `examples/ml_pipeline.py` — scikit-learn TF-IDF + LogisticRegression pipeline with a sample evaluation.
- `examples/kg_node2vec.py` — Node2Vec-style random walks trained with `gensim` Word2Vec.

Run an example:

```powershell
python examples/nlp_normalize.py
python examples/ml_pipeline.py
python examples/kg_node2vec.py
```

These scripts include small built-in example data so they can run without external CSVs.
