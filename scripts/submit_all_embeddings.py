import os
import re
import argparse
from job_utils import run_commands_concurrently

EMB_TAG_RE = re.compile(r"_p_([^_]+)_q_([^_]+)_g_([^_]+)_dim_")


def get_emb_params(emb_file: str) -> tuple[str, str, str]:
    """
    parses an emb_cache/ filename (emb_<edg_tag>_p_<p>_q_<q>_g_<g>_dim_<dim>_
    wl_<wl>_ws_<ws>.tsv, see EmbeddingParams.cache_tag) and returns (p, q, g)
    """
    match = EMB_TAG_RE.search(emb_file)
    if match is None:
        raise ValueError(f"{emb_file!r} doesn't look like an emb_cache/ filename")
    return match.group(1), match.group(2), match.group(3)


def submit_param_jobs(
    params_list: list[tuple[str, str, str]],
    data_dir: str,
    edges_file: str,
    samples_file: str,
    feature_files: list[str],
    out_dir: str,
    max_jobs: int,
) -> None:
    """
    evaluate each (p, q, g) embedding via scripts/sweep.py, running at most
    max_jobs concurrently - --embedding-file points directly at the
    already-generated embedding (skips regenerating it), so this just scores
    every embedding space the sweep already produced.
    """
    cmds = []
    for emb_file, (p, q, g) in params_list:
        cmds.append(
            [
                "python",
                "scripts/sweep.py",
                "--edges-file",
                edges_file,
                "--samples-file",
                samples_file,
                "--feature-files",
                *feature_files,
                "--embedding-file",
                os.path.join(data_dir, emb_file),
                "--p",
                p,
                "--q",
                q,
                "--g",
                g,
                "--save-to",
                out_dir,
                "--no-wandb",
            ]
        )
    run_commands_concurrently(
        commands=cmds,
        max_jobs=max_jobs,
        log_file=os.path.join("logs", "compare_embeddings.log"),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data-dir",
        required=True,
        help="directory of already-generated embeddings (e.g. emb_cache/)",
    )
    p.add_argument("--edges-file", required=True, help="edge list the embeddings were built from")
    p.add_argument(
        "--samples-file",
        required=True,
        help="path to a newline-separated list of sample node IDs",
    )
    p.add_argument(
        "--feature-files",
        required=True,
        nargs="+",
        help="one or more newline-separated lists of feature node IDs",
    )
    p.add_argument(
        "--out", required=True, help="directory to save each embedding's results JSON to"
    )
    p.add_argument(
        "--max_jobs", help="Number of models to train at one time.", type=int, default=4
    )
    args = p.parse_args()

    files = os.listdir(args.data_dir)
    params = [(f, get_emb_params(f)) for f in files]
    submit_param_jobs(
        params_list=params,
        data_dir=args.data_dir,
        edges_file=args.edges_file,
        samples_file=args.samples_file,
        feature_files=args.feature_files,
        out_dir=args.out,
        max_jobs=args.max_jobs,
    )
