#!/usr/bin/env sh

# Usage: ./run/local_10_fit_permutation_test_diet.sh
# Same as run/local_10_fit_permutation_test_time.sh, for diet - no
# --samples-file (diet_labels.tsv only covers endpoint samples, so
# Dataset.from_label_tsv trusts its own rows - see scripts/sweep_setup.py).
D=results/deployment_for_permutations
exec conda run -n multiomics python scripts/fit_permutation_test.py \
    --embeddings data/emb/emb_p_1.0795506927238254_q_8.383911078685804_g_1.tsv.gz \
                 data/emb/emb_p_0.5_q_1.895944090041435_g_1.tsv.gz \
                 data/emb/emb_p_7.305688086564288_q_7.517332462471247_g_2.tsv.gz \
                 data/emb/emb_p_0.8055551041134607_q_0.1_g_1.tsv.gz \
                 data/emb/emb_p_19.0_q_8.483911078685804_g_2.tsv.gz \
                 data/emb/emb_p_5.5_q_9.010757520712524_g_1.tsv.gz \
                 data/emb/emb_p_19.0_q_9.122152261131532_g_1.tsv.gz \
    --model-logs "$D/diet_edges_p_1.0795506927238254_q_8.383911078685804_g_1_dim_128_nw_10_wl_80_ws_10_logging.txt" \
                 "$D/diet_edges_p_0.5_q_1.895944090041435_g_1_dim_128_nw_10_wl_80_ws_10_logging.txt" \
                 "$D/diet_edges_p_7.305688086564288_q_7.517332462471247_g_2_dim_128_nw_10_wl_80_ws_10_logging.txt" \
                 "$D/diet_edges_p_0.8055551041134607_q_0.1_g_1_dim_128_nw_10_wl_80_ws_10_logging.txt" \
                 "$D/diet_edges_p_19.0_q_8.483911078685804_g_2_dim_128_nw_10_wl_80_ws_10_logging.txt" \
                 "$D/diet_edges_p_5.5_q_9.010757520712524_g_1_dim_128_nw_10_wl_80_ws_10_logging.txt" \
                 "$D/diet_edges_p_19.0_q_9.122152261131532_g_1_dim_128_nw_10_wl_80_ws_10_logging.txt" \
    --label-name diet --label-tsv data/diet_labels.tsv --split-tsv data/node_splits.tsv \
    --feature-group microbes=data/nodes/microbes.txt metabolites=data/nodes/metabolites.txt \
    --pred-columns dairy meat --threshold 2.0 --seed 42 \
    --out results/permutations_diet_fit
