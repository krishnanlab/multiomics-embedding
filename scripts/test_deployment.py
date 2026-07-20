"""
Quick manual smoke test of the deployment pipeline: trains one embedding's
time/diet classifiers with a reduced 10-iteration hyperparameter search
(instead of the full 500), writing output to a scratch test directory.
Uses the exact same train_deployment_models.main() as the real pipeline -
just with a smaller search so it runs in seconds instead of minutes.

Usage: python scripts/test_deployment.py [--out results/test_deployment]

"""

from argparse import ArgumentParser

import train_deployment_models as tdm

TEST_TAG = "wcksnlsg"
TEST_PARAMS = {"p": 19.0, "q": 9.122152261131532, "g": 1}
TEST_N_ITER = 10

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--out",
        help="output directory to save files",
        required=False,
        type=str,
        default="results/test_deployment",
    )
    args = parser.parse_args()

    tdm.N_MODELS = TEST_N_ITER  # reduce RandomizedSearchCV iterations for a quick test
    out_dir = tdm.setup_output_dir(args.out)
    tdm.main(
        p=TEST_PARAMS["p"],
        q=TEST_PARAMS["q"],
        g=TEST_PARAMS["g"],
        out_dir=out_dir,
        tag=TEST_TAG,
    )
    print(f"Test run complete: {out_dir}")
