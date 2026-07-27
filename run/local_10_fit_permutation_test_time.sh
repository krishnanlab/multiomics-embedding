#!/usr/bin/env sh

# Usage: ./run/local_10_fit_permutation_test_time.sh
# One-time setup for the time-point permutation test (see
# src/permutation.py): builds a PermutationTest from the 7 curated
# embeddings' already-known best_params (run
# run/local_09_train_deployment_for_permutations.sh first). Reused by
# every permutation-count tier - run once, not once per tier.
D=results/deployment_for_permutations
exec conda run -n multiomics python scripts/fit_permutation_test.py \
    --embeddings data/emb/emb_p_1.0795506927238254_q_8.383911078685804_g_1.tsv.gz \
                 data/emb/emb_p_0.5_q_1.895944090041435_g_1.tsv.gz \
                 data/emb/emb_p_7.305688086564288_q_7.517332462471247_g_2.tsv.gz \
                 data/emb/emb_p_0.8055551041134607_q_0.1_g_1.tsv.gz \
                 data/emb/emb_p_19.0_q_8.483911078685804_g_2.tsv.gz \
                 data/emb/emb_p_5.5_q_9.010757520712524_g_1.tsv.gz \
                 data/emb/emb_p_19.0_q_9.122152261131532_g_1.tsv.gz \
    --model-logs "$D/time_edges_p_1.0795506927238254_q_8.383911078685804_g_1_dim_128_nw_10_wl_80_ws_10_logging.txt" \
                 "$D/time_edges_p_0.5_q_1.895944090041435_g_1_dim_128_nw_10_wl_80_ws_10_logging.txt" \
                 "$D/time_edges_p_7.305688086564288_q_7.517332462471247_g_2_dim_128_nw_10_wl_80_ws_10_logging.txt" \
                 "$D/time_edges_p_0.8055551041134607_q_0.1_g_1_dim_128_nw_10_wl_80_ws_10_logging.txt" \
                 "$D/time_edges_p_19.0_q_8.483911078685804_g_2_dim_128_nw_10_wl_80_ws_10_logging.txt" \
                 "$D/time_edges_p_5.5_q_9.010757520712524_g_1_dim_128_nw_10_wl_80_ws_10_logging.txt" \
                 "$D/time_edges_p_19.0_q_9.122152261131532_g_1_dim_128_nw_10_wl_80_ws_10_logging.txt" \
    --label-name time --label-tsv data/time_labels.tsv --split-tsv data/node_splits.tsv \
    --samples-file data/nodes/samples.txt \
    --feature-group microbes=data/nodes/microbes.txt metabolites=data/nodes/metabolites.txt \
    --pred-columns baseline endpoint --threshold 2.0 --seed 42 \
    --out results/permutations_time_fit
