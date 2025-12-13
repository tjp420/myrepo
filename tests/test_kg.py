from unbreakable_oracle.kg import KnowledgeGraph


def test_kg_entity_and_edge():
    kg = KnowledgeGraph()
    assert kg.add_entity("e1", {"name": "Alice"})
    assert kg.get_entity("e1") == {"name": "Alice"}
    kg.add_entity("e2", {"name": "Bob"})
    kg.add_edge("e1", "e2", "knows")
    assert "e2" in kg.neighbors("e1")
    assert kg.query_by_attr("name", "Alice") == ["e1"]
