import pandas as pd

import src.embedding as embedding_module
from src.embedding import load_or_create_embedding


def build_tiny_edg_file(tmp_path):
    edg_file = tmp_path / "edges.tsv"
    edg_file.write_text("s0\tf0\t1.0\ns1\tf0\t1.0\ns0\tf1\t1.0\ns1\tf1\t1.0\n")
    return edg_file


def make_embedding_kwargs(edg_file, emb_file):
    return dict(
        edg_file=str(edg_file),
        emb_file=str(emb_file),
        n2v_mode="OTF",
        p=1.0, q=1.0, gamma=0.0, seed=0,
        dim=2, num_walks=2, walk_length=3, window_size=2, workers=1,
    )


def test_cache_hit_skips_regeneration_and_returns_identical_data(tmp_path, monkeypatch):
    edg_file = build_tiny_edg_file(tmp_path)
    emb_file = tmp_path / "cache" / "emb.tsv"
    kwargs = make_embedding_kwargs(edg_file, emb_file)

    first = load_or_create_embedding(**kwargs)
    assert emb_file.exists()

    def _fail(*args, **kwargs):
        raise AssertionError("_embed_network should not run on a cache hit")

    monkeypatch.setattr(embedding_module, "_embed_network", _fail)
    second = load_or_create_embedding(**kwargs)

    # dtype float32 (in-memory) vs float64 (round-tripped through text) is
    # expected and harmless - only the column labels/index/values must match
    pd.testing.assert_frame_equal(first, second, check_dtype=False)
