"""
Author: Keenan Manpearl
Date: 2026-07-20

Generic node2vec+ embedding of an edge list, with on-disk caching. Given
edg_file (pecanpy-compatible edge list) and node2vec+ params, returns a
DataFrame embedding indexed by node ID (loaded from emb_file if cached,
else computed and written there). If edg_file is a sample-feature graph,
samples and features land in the same space - which is what lets a
classifier trained on samples also score features.

"""

import os

import pandas as pd
from pecanpy import pecanpy as node2vec


def load_or_create_embedding(
    edg_file: str,
    emb_file: str,
    n2v_mode: str,
    p: float,
    q: float,
    gamma: float,
    seed: int,
    dim: int = 128,
    num_walks: int = 10,
    walk_length: int = 80,
    window_size: int = 10,
    workers: int = 4,
) -> pd.DataFrame:
    """Load emb_file if cached, else compute and cache it. workers should match available CPUs - pecanpy/gensim parallelize via threads, not separate tasks."""
    if os.path.exists(emb_file):
        print(f"Loading embedding from file")
        return pd.read_csv(emb_file, sep="\t", index_col=0)
    os.makedirs(os.path.dirname(emb_file), exist_ok=True)
    return _embed_network(
        edg_file,
        emb_file,
        n2v_mode,
        p,
        q,
        gamma,
        seed,
        dim,
        num_walks,
        walk_length,
        window_size,
        workers,
    )


def _embed_network(
    edg_file: str,
    emb_file: str,
    n2v_mode: str,
    p: float,
    q: float,
    gamma: int,
    seed: int,
    dim: int,
    num_walks: int,
    walk_length: int,
    window_size: int,
    workers: int,
) -> pd.DataFrame:
    """
    load the edge list and create a node2vec+ embedding
    """
    if n2v_mode == "OTF":
        print("Embedding network using SparseOTF")
        g = node2vec.SparseOTF(
            p=p,
            q=q,
            workers=workers,
            verbose=True,
            extend=True,
            gamma=gamma,
            random_state=seed,
        )
        g.read_edg(edg_file, weighted=True, directed=False)
    elif n2v_mode == "Pre":
        print("Embedding network using PreComp")
        g = node2vec.PreComp(
            p=p,
            q=q,
            workers=workers,
            verbose=True,
            extend=True,
            gamma=gamma,
            random_state=seed,
        )
        g.read_edg(edg_file, weighted=True, directed=False)
        g.preprocess_transition_probs()
    nodes = g.nodes
    emb = g.embed(
        dim=dim, num_walks=num_walks, walk_length=walk_length, window_size=window_size
    )
    # match pd.read_csv's string column labels, so a cache-hit and a
    # freshly-computed embedding are interchangeable, not just equal in value
    df = pd.DataFrame(emb, index=nodes, columns=[str(c) for c in range(emb.shape[1])])
    df.to_csv(emb_file, sep="\t")
    return df
