# Knowledge Snippets

This file collects small, runnable example snippets for common tasks (NLP, ML, and
knowledge-graph embeddings). Each snippet includes minimal dependency notes and
how to run it locally.

---

## NLP — spaCy-based normalization and simple KG build

Requirements:
- `pip install pandas spacy`
- `python -m spacy download en_core_web_sm`

```python
import pandas as pd
import spacy

nlp = spacy.load("en_core_web_sm")

df = pd.read_csv("entities.csv")  # columns: text, entity, concept

def normalize_text(text: str) -> str:
    doc = nlp(str(text).lower())
    tokens = [token.lemma_ for token in doc if not token.is_punct and not token.is_space]
    return " ".join(tokens)

df["text_norm"] = df["text"].apply(normalize_text)

# Build a simple knowledge graph (dict of lists)
graph = {}
for _, row in df.iterrows():
    graph.setdefault(row["entity"], []).append(row["concept"])

print("Sample node count:", len(graph))
```

---

## Machine Learning — scikit-learn pipeline + evaluation

Requirements:
- `pip install pandas scikit-learn`

```python
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

data = pd.read_csv("train.csv")  # columns: text, label

X_train, X_test, y_train, y_test = train_test_split(
    data["text"], data["label"], test_size=0.2, random_state=42
)

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1, 2))),
    ("clf", LogisticRegression(max_iter=1000)),
])

pipeline.fit(X_train, y_train)
preds = pipeline.predict(X_test)
print(classification_report(y_test, preds))
```

---

## Knowledge Graph Embeddings — Node2Vec-style using NetworkX + gensim

Requirements:
- `pip install networkx gensim`

```python
import networkx as nx
from gensim.models import Word2Vec
import random

# Build graph from entity-concept pairs or edge list
G = nx.Graph()
G.add_edge("Alice", "MachineLearning")
G.add_edge("Bob", "Databases")
G.add_edge("Alice", "Databases")

def random_walk(G, start, walk_len=10):
    walk = [start]
    for _ in range(walk_len - 1):
        cur = walk[-1]
        neighbors = list(G.neighbors(cur))
        if not neighbors:
            break
        walk.append(random.choice(neighbors))
    return walk

walks = []
for node in G.nodes():
    for _ in range(20):
        walks.append(random_walk(G, node, walk_len=10))

# Train Word2Vec on walks (nodes as tokens)
model = Word2Vec(walks, vector_size=128, window=5, min_count=1, sg=1, epochs=5)

# Example: get embedding vector
vec = model.wv["Alice"]
print("Embedding dimension:", len(vec))
```

---

If you want runnable example scripts under `examples/` and a `requirements.txt` for
these snippets, I can add them and wire a short README with instructions.
