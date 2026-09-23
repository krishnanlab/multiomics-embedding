library(stringr)
library(dplyr)
library(readxl)
library(readr)
library(tidyr)
library(igraph)

############################################################
# Setup
############################################################

out_dir <- paste0(
  "X",
  "Detecting_Local_Communties/classification_association_results/",
  "Clusters_local_communities/Majority_vote"
)

dir.create(
  out_dir,
  recursive = TRUE,
  showWarnings = FALSE
)

thresholds <- c(
  0.50,
  0.60,
  0.70,
  0.80,
  0.90,
  0.95,
  0.975
)

############################################################
# Feature annotation key
############################################################

feature_key_file <- paste0(
  "C:/Users/priceade/Desktop/Fecal_Omics/Older Code and Workflows/",
  "All_micro_metab_KEY.csv"
)

############################################################
# METLIN -> KEGG conversion file
############################################################

metlin_conversion <- read.csv(
  "C:/Users/priceade/Downloads/metlin_id_map_kegg.csv",
  stringsAsFactors = FALSE,
  check.names = FALSE
)

############################################################
# Read feature annotation key
############################################################

feature_key_raw <- read.csv(
  feature_key_file,
  stringsAsFactors = FALSE,
  check.names = FALSE
)

############################################################
# Force relevant columns to simple character vectors
############################################################

feature_key_raw$code <- as.character(
  unlist(
    feature_key_raw$code
  )
)

feature_key_raw$k <- as.character(
  unlist(
    feature_key_raw$k
  )
)

############################################################
# Extract KEGG, METLIN, and HMP IDs from column k
############################################################

feature_key_raw$KEGG_ID <- stringr::str_match(
  feature_key_raw$k,
  "KEGG ID\\s*=\\s*([^,\\]\\s]+)"
)[, 2]

feature_key_raw$METLIN_ID <- stringr::str_match(
  feature_key_raw$k,
  "METLIN ID\\s*=\\s*([^,\\]\\s]+)"
)[, 2]

feature_key_raw$HMP_ID <- stringr::str_match(
  feature_key_raw$k,
  "HMP ID\\s*=\\s*([^,\\]\\s]+)"
)[, 2]

############################################################
# Force IDs to character
############################################################

feature_key_raw <- feature_key_raw %>%
  mutate(
    KEGG_ID = as.character(KEGG_ID),
    METLIN_ID = as.character(METLIN_ID),
    HMP_ID = as.character(HMP_ID)
  )

############################################################
# Prepare METLIN -> KEGG lookup
#
# Query = METLIN ID submitted to converter
# KEGG  = KEGG ID returned by converter
############################################################

metlin_kegg_key <- metlin_conversion %>%
  transmute(
    METLIN_ID = as.character(Query),
    KEGG_from_METLIN = as.character(KEGG)
  ) %>%
  filter(
    !is.na(METLIN_ID),
    METLIN_ID != "",
    !is.na(KEGG_from_METLIN),
    KEGG_from_METLIN != ""
  ) %>%
  distinct(
    METLIN_ID,
    .keep_all = TRUE
  )

############################################################
# Add KEGG IDs recovered from METLIN
############################################################

feature_key_raw <- feature_key_raw %>%
  left_join(
    metlin_kegg_key,
    by = "METLIN_ID"
  ) %>%
  mutate(
    KEGG_ID = coalesce(
      KEGG_ID,
      KEGG_from_METLIN
    )
  )

############################################################
# Show how many KEGG IDs were recovered from METLIN
############################################################

metlin_recovery_summary <- feature_key_raw %>%
  summarise(
    total_features = n(),
    
    original_or_recovered_KEGG = sum(
      !is.na(KEGG_ID) &
        KEGG_ID != ""
    ),
    
    KEGG_recovered_from_METLIN = sum(
      !is.na(KEGG_from_METLIN) &
        KEGG_from_METLIN != ""
    )
  )

print(
  metlin_recovery_summary
)

############################################################
# Optional: inspect recovered mappings
############################################################

recovered_from_METLIN <- feature_key_raw %>%
  filter(
    !is.na(KEGG_from_METLIN),
    KEGG_from_METLIN != ""
  ) %>%
  select(
    code,
    Specie,
    METLIN_ID,
    KEGG_ID,
    KEGG_from_METLIN
  )

print(
  head(
    recovered_from_METLIN,
    50
  )
)

############################################################
# Remove temporary conversion column
############################################################

feature_key_raw <- feature_key_raw %>%
  select(
    -KEGG_from_METLIN
  )

############################################################
# Build final feature lookup
#
# If KEGG exists:
#   N_AQ.1 -> C00695
#
# KEGG may come from:
#   1. original annotation
#   2. METLIN conversion
#
# If KEGG does not exist:
#   retain original feature ID
############################################################

feature_key <- feature_key_raw %>%
  transmute(
    original_feature = as.character(
      code
    ),
    
    KEGG_ID = as.character(
      KEGG_ID
    ),
    
    METLIN_ID = as.character(
      METLIN_ID
    ),
    
    HMP_ID = as.character(
      HMP_ID
    ),
    
    mapped_feature = ifelse(
      !is.na(KEGG_ID) &
        KEGG_ID != "",
      KEGG_ID,
      original_feature
    )
  ) %>%
  distinct(
    original_feature,
    .keep_all = TRUE
  )

############################################################
# Function to map original feature IDs to KEGG
#
# IMPORTANT:
# Used only for enrichment.
# Graph/node identifiers remain unchanged.
############################################################

map_features_to_KEGG <- function(features) {
  
  features <- as.character(
    features
  )
  
  lookup <- data.frame(
    original_feature = features,
    stringsAsFactors = FALSE
  )
  
  lookup <- lookup %>%
    left_join(
      feature_key %>%
        select(
          original_feature,
          mapped_feature
        ),
      by = "original_feature"
    ) %>%
    mutate(
      mapped_feature = ifelse(
        !is.na(mapped_feature) &
          mapped_feature != "",
        mapped_feature,
        original_feature
      )
    )
  
  as.character(
    lookup$mapped_feature
  )
}

############################################################
# Inspect mapping table
############################################################

print(
  head(
    feature_key,
    20
  )
)

############################################################
# Read similarity matrices
############################################################

class_dir <- paste0(
  "X",
  "Detecting_Local_Communties/classification_association_results/Majority_vote"
)

avg_mats <- list(
  
  meat = as.matrix(
    read.csv(
      file.path(
        class_dir,
        "avg_sig_type_mat_meat.csv"
      ),
      row.names = 1,
      check.names = FALSE
    )
  ),
  
  dairy = as.matrix(
    read.csv(
      file.path(
        class_dir,
        "avg_sig_type_mat_dairy.csv"
      ),
      row.names = 1,
      check.names = FALSE
    )
  ),
  
  base = as.matrix(
    read.csv(
      file.path(
        class_dir,
        "avg_sig_type_mat_base.csv"
      ),
      row.names = 1,
      check.names = FALSE
    )
  ),
  
  end = as.matrix(
    read.csv(
      file.path(
        class_dir,
        "avg_sig_type_mat_end.csv"
      ),
      row.names = 1,
      check.names = FALSE
    )
  )
)

############################################################
# Ensure similarity matrices are numeric
############################################################

avg_mats <- lapply(
  avg_mats,
  function(x) {
    
    storage.mode(x) <- "numeric"
    
    x
  }
)

############################################################
# Pull feature names used in classification analysis
############################################################

pecan_matrix <- read.table(
  file = paste0(
    "X",
    "top_embeddings/emb_p_0.5_q_1.895944090041435_g_1.tsv.gz"
  ),
  fill = TRUE
)

colnames(
  pecan_matrix
) <- paste0(
  "dim_",
  1:128
)

pecan_matrix_128 <- pecan_matrix

pecan_matrix_128$nodes <- rownames(
  pecan_matrix_128
)

pecan_matrix_128 <- pecan_matrix_128[
  ,
  c(
    129,
    1:128
  )
]

############################################################
# Original embedding node IDs
############################################################

used_features_original <- as.character(
  pecan_matrix_128$nodes
)

############################################################
# Map enrichment universe to KEGG where available
############################################################

used_features <- map_features_to_KEGG(
  used_features_original
)

############################################################
# Save/check embedding feature mapping
############################################################

used_features_df <- data.frame(
  original_feature = used_features_original,
  feature = used_features,
  stringsAsFactors = FALSE
)

used_features_df <- used_features_df %>%
  left_join(
    feature_key %>%
      select(
        original_feature,
        KEGG_ID,
        METLIN_ID,
        HMP_ID
      ),
    by = "original_feature"
  )

embedding_mapping_summary <- used_features_df %>%
  summarise(
    total_features = n(),
    
    converted_to_KEGG = sum(
      feature != original_feature,
      na.rm = TRUE
    ),
    
    original_retained = sum(
      feature == original_feature,
      na.rm = TRUE
    )
  )

print(
  embedding_mapping_summary
)

############################################################
# Read and prepare KEGG annotations
############################################################

KEGG_reference <- read_xlsx(
  paste0(
    "X",
    "Enrichment_Analysis/kegg_pathway_feature_mapping_full.xlsx"
  )
)

KEGG_levels <- as.data.frame(
  read.csv(
    paste0(
      "X",
      "Enrichment_Analysis/Parsed_KEGG_Full_Hierarchy.csv"
    ),
    stringsAsFactors = FALSE
  )
)

KEGG_reference <- as.data.frame(
  KEGG_reference
)

############################################################
# Ensure matching variables are character
############################################################

KEGG_reference$feature <- as.character(
  KEGG_reference$feature
)

KEGG_reference$details <- as.character(
  KEGG_reference$details
)

KEGG_levels$l1 <- as.character(
  KEGG_levels$l1
)

KEGG_levels$l2 <- as.character(
  KEGG_levels$l2
)

KEGG_levels$l3 <- as.character(
  KEGG_levels$l3
)

############################################################
# Create KEGG hierarchy-level annotation tables
############################################################

KEGG_l1 <- KEGG_reference %>%
  filter(
    details %in%
      KEGG_levels$l1
  )

KEGG_l2 <- KEGG_reference %>%
  filter(
    details %in%
      KEGG_levels$l2
  )

not_l1 <- KEGG_reference %>%
  filter(
    !details %in%
      KEGG_levels$l1
  )

not_l2 <- KEGG_reference %>%
  filter(
    !details %in%
      KEGG_levels$l2
  )

KEGG_l3 <- semi_join(
  not_l1,
  not_l2,
  by = "details"
)

############################################################
# Restrict KEGG annotations to features used in analysis
############################################################

used_KEGG <- list(
  
  L1 = KEGG_l1 %>%
    filter(
      feature %in%
        used_features
    ),
  
  L2 = KEGG_l2 %>%
    filter(
      feature %in%
        used_features
    ),
  
  L3 = KEGG_l3 %>%
    filter(
      feature %in%
        used_features
    )
)

############################################################
# Summarize annotation universe
############################################################

used_KEGG_summary <- data.frame(
  
  KEGG_level = names(
    used_KEGG
  ),
  
  annotation_rows = sapply(
    used_KEGG,
    nrow
  ),
  
  unique_features = sapply(
    used_KEGG,
    function(x) {
      
      length(
        unique(
          as.character(
            x$feature
          )
        )
      )
    }
  )
)

print(
  used_KEGG_summary
)

############################################################
# KEGG enrichment function
############################################################

run_KEGG_enrichment <- function(
    cluster_features,
    background_features,
    KEGG_level
) {
  
  if(
    !KEGG_level %in%
    names(
      used_KEGG
    )
  ) {
    
    stop(
      "KEGG_level must be L1, L2, or L3."
    )
  }
  
  ##########################################################
  # Map graph node IDs to KEGG where available
  ##########################################################
  
  cluster_features_mapped <- map_features_to_KEGG(
    cluster_features
  )
  
  background_features_mapped <- map_features_to_KEGG(
    background_features
  )
  
  ##########################################################
  # Remove duplicated mapped IDs
  ##########################################################
  
  cluster_features_mapped <- unique(
    as.character(
      cluster_features_mapped
    )
  )
  
  background_features_mapped <- unique(
    as.character(
      background_features_mapped
    )
  )
  
  ##########################################################
  # Pull appropriate KEGG annotation table
  ##########################################################
  
  KEGG_table <- used_KEGG[[KEGG_level]]
  
  KEGG_table <- KEGG_table %>%
    filter(
      !is.na(feature),
      feature != "",
      !is.na(details),
      details != ""
    ) %>%
    distinct(
      feature,
      details,
      .keep_all = TRUE
    )
  
  ##########################################################
  # Cluster annotation rows
  ##########################################################
  
  cluster_KEGG <- KEGG_table %>%
    filter(
      feature %in%
        cluster_features_mapped
    )
  
  ##########################################################
  # Full network background annotation rows
  ##########################################################
  
  background_KEGG <- KEGG_table %>%
    filter(
      feature %in%
        background_features_mapped
    )
  
  ##########################################################
  # Non-cluster annotation rows
  ##########################################################
  
  noncluster_KEGG <- background_KEGG %>%
    filter(
      !feature %in%
        cluster_features_mapped
    )
  
  ##########################################################
  # Skip empty comparisons
  ##########################################################
  
  if(
    nrow(cluster_KEGG) == 0 ||
    nrow(noncluster_KEGG) == 0
  ) {
    
    return(
      NULL
    )
  }
  
  ##########################################################
  # KEGG terms observed in cluster
  ##########################################################
  
  KEGG_terms <- sort(
    unique(
      as.character(
        cluster_KEGG$details
      )
    )
  )
  
  KEGG_terms <- KEGG_terms[
    !is.na(KEGG_terms) &
      KEGG_terms != ""
  ]
  
  if(
    length(
      KEGG_terms
    ) == 0
  ) {
    
    return(
      NULL
    )
  }
  
  ##########################################################
  # Run enrichment tests
  ##########################################################
  
  enrichment_results <- lapply(
    KEGG_terms,
    function(term) {
      
      cluster_fn <- sum(
        cluster_KEGG$details ==
          term
      )
      
      cluster_notfn <-
        nrow(
          cluster_KEGG
        ) -
        cluster_fn
      
      noncluster_fn <- sum(
        noncluster_KEGG$details ==
          term
      )
      
      noncluster_notfn <-
        nrow(
          noncluster_KEGG
        ) -
        noncluster_fn
      
      ######################################################
      # Fisher exact test
      ######################################################
      
      test <- fisher.test(
        matrix(
          c(
            cluster_fn,
            noncluster_fn,
            cluster_notfn,
            noncluster_notfn
          ),
          nrow = 2,
          ncol = 2
        ),
        alternative = "greater"
      )
      
      ######################################################
      # Proportions
      ######################################################
      
      cluster_proportion <-
        cluster_fn /
        nrow(
          cluster_KEGG
        )
      
      background_count <-
        cluster_fn +
        noncluster_fn
      
      background_total <-
        nrow(
          cluster_KEGG
        ) +
        nrow(
          noncluster_KEGG
        )
      
      background_proportion <-
        background_count /
        background_total
      
      fold_enrichment <- ifelse(
        background_proportion > 0,
        
        cluster_proportion /
          background_proportion,
        
        NA_real_
      )
      
      ######################################################
      # Result row
      ######################################################
      
      data.frame(
        KEGG_level =
          KEGG_level,
        
        KEGG_category =
          term,
        
        cluster_count =
          cluster_fn,
        
        cluster_total =
          nrow(
            cluster_KEGG
          ),
        
        noncluster_count =
          noncluster_fn,
        
        noncluster_total =
          nrow(
            noncluster_KEGG
          ),
        
        background_count =
          background_count,
        
        background_total =
          background_total,
        
        cluster_proportion =
          cluster_proportion,
        
        background_proportion =
          background_proportion,
        
        fold_enrichment =
          fold_enrichment,
        
        p_value =
          test$p.value,
        
        stringsAsFactors = FALSE
      )
    }
  )
  
  ##########################################################
  # Combine results
  ##########################################################
  
  enrich_df <- bind_rows(
    enrichment_results
  )
  
  ##########################################################
  # Remove uninformative terms
  ##########################################################
  
  enrich_df <- enrich_df %>%
    filter(
      !grepl(
        "Function unknown",
        KEGG_category,
        ignore.case = TRUE
      ),
      
      !grepl(
        "General function prediction only",
        KEGG_category,
        ignore.case = TRUE
      )
    )
  
  ##########################################################
  # Require at least four annotation rows in cluster
  ##########################################################
  
  enrich_df <- enrich_df %>%
    filter(
      cluster_count >= 4
    )
  
  if(
    nrow(
      enrich_df
    ) == 0
  ) {
    
    return(
      NULL
    )
  }
  
  ##########################################################
  # FDR correction
  ##########################################################
  
  enrich_df <- enrich_df %>%
    mutate(
      FDR = p.adjust(
        p_value,
        method = "fdr",
        n = length(
          p_value
        )
      )
    ) %>%
    arrange(
      FDR,
      p_value,
      desc(
        fold_enrichment
      )
    )
  
  return(
    enrich_df
  )
}

############################################################
# Run thresholds, Louvain clustering, and KEGG enrichment
############################################################

all_louvain_summaries <- list()

all_enrichment_results <- list()

all_membership_results <- list()

set.seed(
  10
)

for(
  thresh in thresholds
) {
  
  message(
    "Running threshold: ",
    thresh
  )
  
  edge_list_results <- list()
  
  louvain_results <- list()
  
  louvain_memberships <- list()
  
  ##########################################################
  # Create edge lists for threshold
  ##########################################################
  
  for(
    type in names(
      avg_mats
    )
  ) {
    
    sim_mat <- avg_mats[[type]]
    
    if(
      is.null(
        rownames(
          sim_mat
        )
      ) ||
      is.null(
        colnames(
          sim_mat
        )
      )
    ) {
      
      stop(
        paste0(
          "Similarity matrix ",
          type,
          " must have row and column names."
        )
      )
    }
    
    if(
      !identical(
        rownames(
          sim_mat
        ),
        colnames(
          sim_mat
        )
      )
    ) {
      
      stop(
        paste0(
          "Row and column names do not match for network: ",
          type
        )
      )
    }
    
    ########################################################
    # Remove diagonal
    ########################################################
    
    diag(
      sim_mat
    ) <- 0
    
    ########################################################
    # Upper triangle
    ########################################################
    
    edges_tmp <- data.frame(
      
      Source = rownames(
        sim_mat
      )[
        row(
          sim_mat
        )[
          upper.tri(
            sim_mat
          )
        ]
      ],
      
      Target = colnames(
        sim_mat
      )[
        col(
          sim_mat
        )[
          upper.tri(
            sim_mat
          )
        ]
      ],
      
      width = sim_mat[
        upper.tri(
          sim_mat
        )
      ],
      
      stringsAsFactors = FALSE
    )
    
    edges_tmp <- edges_tmp %>%
      filter(
        is.finite(
          width
        )
      )
    
    if(
      nrow(
        edges_tmp
      ) == 0
    ) {
      
      warning(
        paste0(
          "No finite similarities for ",
          type,
          " at threshold ",
          thresh
        )
      )
      
      next
    }
    
    ########################################################
    # Threshold value
    ########################################################
    
    threshold_value <- quantile(
      edges_tmp$width,
      probs = thresh,
      na.rm = TRUE,
      names = FALSE
    )
    
    sim_mat[
      is.na(
        sim_mat
      ) |
        sim_mat <
        threshold_value
    ] <- 0
    
    ########################################################
    # Square retained similarities
    ########################################################
    
    sim_mat <- sim_mat^2
    
    ########################################################
    # Build retained edge list
    ########################################################
    
    edges <- data.frame(
      
      Source = rownames(
        sim_mat
      )[
        row(
          sim_mat
        )[
          upper.tri(
            sim_mat
          )
        ]
      ],
      
      Target = colnames(
        sim_mat
      )[
        col(
          sim_mat
        )[
          upper.tri(
            sim_mat
          )
        ]
      ],
      
      width = sim_mat[
        upper.tri(
          sim_mat
        )
      ],
      
      stringsAsFactors = FALSE
    )
    
    edges <- edges %>%
      filter(
        is.finite(
          width
        ),
        width > 0
      )
    
    edge_list_results[[type]] <-
      edges
  }
  
  ##########################################################
  # Louvain clustering
  ##########################################################
  
  threshold_summary_list <- list()
  
  for(
    type in names(
      edge_list_results
    )
  ) {
    
    edges <- edge_list_results[[type]]
    
    if(
      nrow(
        edges
      ) == 0
    ) {
      
      warning(
        paste0(
          "No retained edges for ",
          type,
          " at threshold ",
          thresh
        )
      )
      
      next
    }
    
    ########################################################
    # Build graph
    ########################################################
    
    g <- graph_from_data_frame(
      d = edges,
      directed = FALSE
    )
    
    ########################################################
    # Louvain
    ########################################################
    
    louvain_fit <- cluster_louvain(
      graph = g,
      weights = E(g)$width
    )
    
    louvain_results[[type]] <-
      louvain_fit
    
    memb <- membership(
      louvain_fit
    )
    
    cluster_sizes <- table(
      memb
    )
    
    ########################################################
    # Summary
    ########################################################
    
    threshold_summary_list[[type]] <-
      data.frame(
        
        threshold =
          thresh,
        
        network =
          type,
        
        n_nodes =
          vcount(
            g
          ),
        
        n_edges =
          ecount(
            g
          ),
        
        n_clusters =
          length(
            cluster_sizes
          ),
        
        modularity = modularity(
          g,
          membership = memb,
          weights = E(g)$width
        ),
        
        largest_cluster =
          as.numeric(
            max(
              cluster_sizes
            )
          ),
        
        smallest_cluster =
          as.numeric(
            min(
              cluster_sizes
            )
          ),
        
        mean_cluster_size =
          round(
            mean(
              as.numeric(
                cluster_sizes
              )
            ),
            2
          ),
        
        median_cluster_size =
          round(
            median(
              as.numeric(
                cluster_sizes
              )
            ),
            2
          )
      )
    
    ########################################################
    # Membership dataframe
    ########################################################
    
    membership_df <- data.frame(
      
      feature =
        names(
          memb
        ),
      
      cluster =
        as.numeric(
          memb
        ),
      
      network =
        type,
      
      threshold =
        thresh,
      
      stringsAsFactors = FALSE
    )
    
    louvain_memberships[[type]] <-
      membership_df
    
    all_membership_results[[
      paste(
        thresh,
        type,
        sep = "_"
      )
    ]] <- membership_df
  }
  
  ##########################################################
  # Save threshold summaries
  ##########################################################
  
  if(
    length(
      threshold_summary_list
    ) > 0
  ) {
    
    all_louvain_summaries[[
      as.character(
        thresh
      )
    ]] <- bind_rows(
      threshold_summary_list
    )
  }
  
  ##########################################################
  # KEGG enrichment
  ##########################################################
  
  for(
    type in names(
      louvain_memberships
    )
  ) {
    
    membership_df <-
      louvain_memberships[[type]]
    
    background_features <-
      as.character(
        membership_df$feature
      )
    
    for(
      clust in sort(
        unique(
          membership_df$cluster
        )
      )
    ) {
      
      cluster_features <- membership_df %>%
        filter(
          cluster == clust
        ) %>%
        pull(
          feature
        ) %>%
        as.character()
      
      for(
        KEGG_level in c(
          "L1",
          "L2",
          "L3"
        )
      ) {
        
        enrich_df <- run_KEGG_enrichment(
          
          cluster_features =
            cluster_features,
          
          background_features =
            background_features,
          
          KEGG_level =
            KEGG_level
        )
        
        if(
          !is.null(
            enrich_df
          ) &&
          nrow(
            enrich_df
          ) > 0
        ) {
          
          enrich_df <- enrich_df %>%
            mutate(
              
              threshold =
                thresh,
              
              network =
                type,
              
              cluster =
                clust,
              
              n_features_in_cluster =
                length(
                  cluster_features
                ),
              
              n_mapped_features_in_cluster =
                length(
                  unique(
                    map_features_to_KEGG(
                      cluster_features
                    )
                  )
                ),
              
              .before = 1
            ) %>%
            select(
              
              threshold,
              network,
              cluster,
              n_features_in_cluster,
              n_mapped_features_in_cluster,
              KEGG_level,
              KEGG_category,
              cluster_count,
              cluster_total,
              noncluster_count,
              noncluster_total,
              background_count,
              background_total,
              cluster_proportion,
              background_proportion,
              fold_enrichment,
              p_value,
              FDR
            )
          
          result_name <- paste(
            thresh,
            type,
            clust,
            KEGG_level,
            sep = "_"
          )
          
          all_enrichment_results[[
            result_name
          ]] <- enrich_df
        }
      }
    }
  }
}

############################################################
# Combine Louvain summaries
############################################################

if(
  length(
    all_louvain_summaries
  ) > 0
) {
  
  louvain_threshold_summary <- bind_rows(
    all_louvain_summaries
  )
  
  rownames(
    louvain_threshold_summary
  ) <- NULL
  
  write.csv(
    louvain_threshold_summary,
    file.path(
      out_dir,
      "louvain_threshold_summary_KEGG.csv"
    ),
    row.names = FALSE
  )
  
  print(
    louvain_threshold_summary
  )
  
} else {
  
  warning(
    "No Louvain summaries were generated."
  )
}

############################################################
# Combine cluster memberships
############################################################

if(
  length(
    all_membership_results
  ) > 0
) {
  
  all_louvain_memberships <- bind_rows(
    all_membership_results
  )
  
  rownames(
    all_louvain_memberships
  ) <- NULL
  
  ##########################################################
  # Add KEGG-mapped feature for reference
  ##########################################################
  
  all_louvain_memberships <- all_louvain_memberships %>%
    mutate(
      KEGG_feature =
        map_features_to_KEGG(
          feature
        )
    )
  
  write.csv(
    all_louvain_memberships,
    file.path(
      out_dir,
      "all_threshold_network_louvain_memberships_KEGG.csv"
    ),
    row.names = FALSE
  )
}

############################################################
# Combine KEGG enrichment results
############################################################

if(
  length(
    all_enrichment_results
  ) > 0
) {
  
  all_KEGG_enrichment_results <- bind_rows(
    all_enrichment_results
  )
  
  rownames(
    all_KEGG_enrichment_results
  ) <- NULL
  
  write.csv(
    all_KEGG_enrichment_results,
    file.path(
      out_dir,
      "all_threshold_cluster_KEGG_enrichment_results.csv"
    ),
    row.names = FALSE
  )
  
  ##########################################################
  # Separate files for each KEGG level
  ##########################################################
  
  for(
    current_KEGG_level in c(
      "L1",
      "L2",
      "L3"
    )
  ) {
    
    level_results <- all_KEGG_enrichment_results %>%
      filter(
        KEGG_level ==
          current_KEGG_level
      )
    
    write.csv(
      level_results,
      file.path(
        out_dir,
        paste0(
          "all_threshold_cluster_KEGG_",
          current_KEGG_level,
          "_enrichment_results.csv"
        )
      ),
      row.names = FALSE
    )
  }
  
  ##########################################################
  # Top enrichment per threshold/network/cluster/level
  ##########################################################
  
  top_KEGG_enrichments <- all_KEGG_enrichment_results %>%
    group_by(
      threshold,
      network,
      cluster,
      KEGG_level
    ) %>%
    arrange(
      FDR,
      p_value,
      desc(
        fold_enrichment
      ),
      .by_group = TRUE
    ) %>%
    slice_head(
      n = 5
    ) %>%
    ungroup()
  
  write.csv(
    top_KEGG_enrichments,
    file.path(
      out_dir,
      paste0(
        "top_KEGG_enrichments_by_threshold_",
        "network_cluster_level.csv"
      )
    ),
    row.names = FALSE
  )
  
  print(
    top_KEGG_enrichments
  )
  
  ##########################################################
  # Significant KEGG enrichment results
  ##########################################################
  
  significant_KEGG_enrichments <- all_KEGG_enrichment_results %>%
    filter(
      FDR < 0.05
    ) %>%
    arrange(
      threshold,
      network,
      cluster,
      KEGG_level,
      FDR,
      p_value
    )
  
  write.csv(
    significant_KEGG_enrichments,
    file.path(
      out_dir,
      paste0(
        "significant_threshold_cluster_",
        "KEGG_enrichments_FDR_0.05.csv"
      )
    ),
    row.names = FALSE
  )
  
} else {
  
  warning(
    paste0(
      "No KEGG enrichment results passed the ",
      "minimum cluster count of four."
    )
  )
}

############################################################
# Save final feature mapping lookup
#
# Includes KEGG IDs recovered through METLIN
############################################################

write.csv(
  feature_key,
  file.path(
    out_dir,
    "original_feature_to_KEGG_lookup.csv"
  ),
  row.names = FALSE
)

############################################################
# Save full raw feature key with parsed IDs
############################################################

write.csv(
  feature_key_raw,
  file.path(
    out_dir,
    "feature_key_with_KEGG_METLIN_HMP_ids.csv"
  ),
  row.names = FALSE
)

############################################################
# Save embedding/background mapping
############################################################

write.csv(
  used_features_df,
  file.path(
    out_dir,
    "embedding_feature_to_KEGG_lookup.csv"
  ),
  row.names = FALSE
)