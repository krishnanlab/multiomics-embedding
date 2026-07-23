"""
Generic node2vec+ embedding of an edge list, with on-disk caching.

Not specific to any one study: takes an edge-list file and node2vec+
parameters, and returns (loading from a cached file if one already exists at
emb_file, otherwise computing and writing one) a DataFrame embedding, indexed
by node ID, with one column per embedding dimension. If the input edge list
is a sample-feature graph, the resulting embedding jointly places samples and
features in the same space - which is what lets a classifier trained on
sample rows also score feature rows.

Required data format
---------------------
- edg_file: a pecanpy-compatible edge list (whitespace/tab-separated
  "node1 node2 weight" per line).

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
    """
    Load a cached embedding from emb_file if it exists; otherwise create one
    from edg_file with the given node2vec+ parameters and cache it there.
    dim/num_walks/walk_length/window_size default to pecanpy's own library
    defaults. workers should match however many CPUs are actually available
    (e.g. a SLURM job's --cpus-per-task) - pecanpy/gensim parallelize within
    this one process via threads, not separate tasks.
    """
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
    df = pd.DataFrame(emb, index=nodes)
    df.to_csv(emb_file, sep="\t")
    return df
