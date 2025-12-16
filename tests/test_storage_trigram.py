import pytest
from sigil_mcp.storage.trigram import TrigramIndex, ROCKSDICT_AVAILABLE, ROCKSDB_AVAILABLE


def test_trigram_index_basic(tmp_path):
    if not (ROCKSDICT_AVAILABLE or ROCKSDB_AVAILABLE):
        pytest.skip("No RocksDB backend available for trigram index")
    p = tmp_path / "trigrams.rocksdb"
    t = TrigramIndex(p)
    # empty initially
    assert t.count() == 0
    # set and get
    t.set_doc_ids("abc", {1, 2, 3})
    assert t.get_doc_ids("abc") == {1, 2, 3}
    # delete
    t.delete("abc")
    assert t.get_doc_ids("abc") == set()
    t.close()


def test_trigram_serialize_roundtrip():
    s = {"abc", "def\n", "g"}
    from sigil_mcp.storage.trigram import TrigramIndex

    b = TrigramIndex.serialize_trigram_set(s)
    out = TrigramIndex.deserialize_trigram_set(b)
    assert s == out


def test_serialize_doc_ids_roundtrip():
    # Static helpers should round-trip doc id sets
    from sigil_mcp.storage.trigram import TrigramIndex

    inp = {1, 42, 999}
    b = TrigramIndex._serialize_doc_ids(inp)
    out = TrigramIndex._deserialize_doc_ids(b)
    assert inp == out


def test_serialize_trigram_unicode_and_empty():
    from sigil_mcp.storage.trigram import TrigramIndex

    s = {"", "αβγ", "line\nwith\nnewline"}
    b = TrigramIndex.serialize_trigram_set(s)
    out = TrigramIndex.deserialize_trigram_set(b)
    assert s == out
