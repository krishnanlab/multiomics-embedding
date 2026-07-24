"""
Author: Keenan Manpearl
Date: 2026-07-22

Generic node-list validation against a graph, and optionally an embedding.
Not study-specific: works on any node ID lists, edge list, or embedding.

Reuses src/dataset.py's read_ids/require_present/warn_if_unaccounted (same
"error if missing, warn if unclaimed" pattern as Dataset.from_label_tsv) -
the only new piece here is checking against a *graph* (edge list).

"""

from src.dataset import read_ids, require_present, warn_if_unaccounted


def read_graph_nodes(edg_file: str) -> set[str]:
    """Unique node IDs referenced by an edge list - a lightweight line-parse, not a full graph load."""
    nodes: set[str] = set()
    with open(edg_file) as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 2:
                nodes.add(parts[0])
                nodes.add(parts[1])
    return nodes


def validate_node_lists(
    edg_file: str,
    list_files: dict[str, str],
    embedding_nodes: set[str] | None = None,
) -> None:
    """
    list_files: {name: path}, one ID per line each. Raises ValueError if a
    listed node is missing from the graph or embedding_nodes; warns if the
    graph/embedding has nodes not named in any list.
    """
    graph_nodes = read_graph_nodes(edg_file)
    all_listed: set[str] = set()

    for name, path in list_files.items():
        listed = read_ids(path)
        all_listed |= listed
        require_present(listed, graph_nodes, name, "a row in the graph")
        if embedding_nodes is not None:
            require_present(listed, embedding_nodes, name, "a row in the embedding")

    warn_if_unaccounted(graph_nodes, all_listed, "graph nodes")
    if embedding_nodes is not None:
        warn_if_unaccounted(embedding_nodes, all_listed, "embedding rows")
