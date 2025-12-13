"""Simple Node2Vec-style example using NetworkX + gensim.

Usage:
    python examples/kg_node2vec.py

This script runs a small random-walk corpus and trains a Word2Vec model over node walks.
"""
import random


def main():
    try:
        import networkx as nx
        from gensim.models import Word2Vec
    except Exception:
        print("Missing dependencies. Install with: pip install networkx gensim")
        raise

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

    model = Word2Vec(walks, vector_size=128, window=5, min_count=1, sg=1, epochs=5)
    print("Trained Word2Vec over", len(walks), "walks. Sample vector dim:", len(model.wv["Alice"]))

if __name__ == "__main__":
    main()
