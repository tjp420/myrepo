from unbreakable_oracle.utils import chunk_text, clean_text


def test_clean_text():
    assert clean_text("  a   b\n") == "a b"


def test_chunk_text_short():
    assert chunk_text("short", chunk_size=10) == ["short"]


def test_chunk_text_long():
    s = "a" * 200
    chunks = chunk_text(s, chunk_size=50, overlap=10)
    assert len(chunks) > 1
    # chunks should cover most of the original text
    joined = "".join(chunks)
    assert len(joined) >= len(s) - 50
