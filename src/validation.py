"""
Generic node-list validation against a graph, and optionally against an
embedding. Not specific to any one study: works on any node ID lists (not
just "samples"/"features"), any edge list, and any embedding.

Reuses src/dataset.py's read_ids/require_present/warn_if_unaccounted -
those already implement this exact "error if a listed node is missing,
warn if a target has unlisted nodes" pattern for validating a label/
feature list against an embedding (see Dataset.from_label_tsv). The only
thing genuinely new here is doing the same check against a *graph* (an
edge list), which from_label_tsv has no reason to know about.

"""

from src.dataset import read_ids, require_present, warn_if_unaccounted


def read_graph_nodes(edg_file: str) -> set[str]:
    """
    unique node IDs referenced by an edge list (whitespace/tab-separated
    "node1 node2 weight" per line) - a lightweight line-parse, not a full
    graph load, since only node identity is needed here.
    """
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
    embedding_nodes: "set[str] | None" = None,
) -> None:
    """
    list_files: {name: path}, e.g. {"samples": ..., "microbes": ...,
    "metabolites": ...} - one ID per line each. Raises ValueError if any
    listed node is missing from the graph, or from embedding_nodes when
    given. Warns (does not raise) if the graph, or the embedding when
    given, contains nodes not named in any list.
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
