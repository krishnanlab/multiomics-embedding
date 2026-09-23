library(stringr)
library(dplyr)
library(readr)
library(tidyr)
library(igraph)

############################################################
# Setup
############################################################

out_dir <- paste0(
  "X",
  "Detecting_Local_Communties/classification_association_results/",
  "Clusters_local_communities/Permutations"
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
# Read similarity matrices
############################################################

class_dir <- paste0(
  "X",
  "Detecting_Local_Communties/classification_association_results/Permutations"
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
# Pull in feature names used in classification analysis
############################################################

pecan_matrix <- read.table(
  file = paste0(
    "X",
    "top_embeddings/emb_p_0.5_q_1.895944090041435_g_1.tsv.gz"
  ),
  fill = TRUE
)

colnames(pecan_matrix) <- paste0(
  "dim_",
  1:128
)

pecan_matrix_128 <- pecan_matrix

pecan_matrix_128$nodes <- rownames(
  pecan_matrix_128
)

pecan_matrix_128 <- pecan_matrix_128[
  ,
  c(129, 1:128)
]

used_codes <- as.character(
  pecan_matrix_128$nodes
)

############################################################
# Read and clean species taxonomy annotations
############################################################

SPE <- read.csv(
  paste0(
    "C:/Users/priceade/Desktop/Fecal_Omics/Older Code and Workflows/",
    "processed_data_1/encoding_species.csv"
  ),
  stringsAsFactors = FALSE,
  check.names = FALSE
)

############################################################
# Standardize species names to embedding format
############################################################

SPE$Specie <- as.character(
  SPE$Specie
)

SPE$Specie <- gsub(
  ";",
  ".",
  SPE$Specie
)

SPE$Specie <- gsub(
  " ",
  ".",
  SPE$Specie
)

SPE$Specie <- gsub(
  "\\(",
  ".",
  SPE$Specie
)

SPE$Specie <- gsub(
  "\\)",
  ".",
  SPE$Specie
)

SPE$Specie <- gsub(
  "\\[",
  ".",
  SPE$Specie
)

SPE$Specie <- gsub(
  "\\]",
  ".",
  SPE$Specie
)

SPE$Specie <- gsub(
  "-",
  ".",
  SPE$Specie
)

############################################################
# Collapse repeated periods and trim ends
############################################################

SPE$Specie <- gsub(
  "\\.+",
  ".",
  SPE$Specie
)

SPE$Specie <- gsub(
  "^\\.",
  "",
  SPE$Specie
)

SPE$Specie <- gsub(
  "\\.$",
  "",
  SPE$Specie
)

############################################################
# Restrict taxonomy universe to features used in analysis
############################################################

used_SPE <- SPE %>%
  filter(
    Specie %in% used_codes
  )

############################################################
# Confirm required taxonomy columns exist
#
# c = class
# o = origin
# f = family
# g = genus
############################################################

required_taxonomy_columns <- c(
  "Specie",
  "f",
  "g",
  "c",
  "o"
)

missing_taxonomy_columns <- setdiff(
  required_taxonomy_columns,
  colnames(
    used_SPE
  )
)

if(
  length(
    missing_taxonomy_columns
  ) > 0
) {
  
  stop(
    paste0(
      "Missing required columns in encoding_species.csv: ",
      paste(
        missing_taxonomy_columns,
        collapse = ", "
      )
    )
  )
}

############################################################
# Force taxonomy columns to character
############################################################

used_SPE <- used_SPE %>%
  mutate(
    Specie = as.character(Specie),
    f = as.character(f),
    g = as.character(g),
    c = as.character(c),
    o = as.character(o)
  )

############################################################
# Taxonomy-level mapping
#
# NEW:
# origin is mapped to column o
############################################################

taxonomy_levels <- c(
  family = "f",
  genus = "g",
  class = "c",
  origin = "o"
)

############################################################
# Function for taxonomy enrichment for one cluster and level
############################################################

run_species_enrichment <- function(
    cluster_features,
    background_features,
    taxonomy_level
) {
  
  ##########################################################
  # Validate taxonomy level
  ##########################################################
  
  if(
    !taxonomy_level %in%
    names(
      taxonomy_levels
    )
  ) {
    
    stop(
      paste0(
        "taxonomy_level must be family, genus, class, ",
        "or origin."
      )
    )
  }
  
  ##########################################################
  # Match level to taxonomy column
  ##########################################################
  
  taxonomy_column <- taxonomy_levels[[taxonomy_level]]
  
  ##########################################################
  # Species annotations inside cluster
  ##########################################################
  
  cluster_SPE <- used_SPE %>%
    filter(
      Specie %in%
        cluster_features
    ) %>%
    transmute(
      feature =
        as.character(
          Specie
        ),
      
      taxonomy_category =
        as.character(
          .data[[taxonomy_column]]
        )
    ) %>%
    mutate(
      feature =
        str_trim(
          feature
        ),
      
      taxonomy_category =
        str_trim(
          taxonomy_category
        )
    ) %>%
    filter(
      !is.na(
        taxonomy_category
      ),
      taxonomy_category != ""
    ) %>%
    distinct(
      feature,
      taxonomy_category
    )
  
  ##########################################################
  # Species annotations in full network background
  ##########################################################
  
  background_SPE <- used_SPE %>%
    filter(
      Specie %in%
        background_features
    ) %>%
    transmute(
      feature =
        as.character(
          Specie
        ),
      
      taxonomy_category =
        as.character(
          .data[[taxonomy_column]]
        )
    ) %>%
    mutate(
      feature =
        str_trim(
          feature
        ),
      
      taxonomy_category =
        str_trim(
          taxonomy_category
        )
    ) %>%
    filter(
      !is.na(
        taxonomy_category
      ),
      taxonomy_category != ""
    ) %>%
    distinct(
      feature,
      taxonomy_category
    )
  
  ##########################################################
  # Species annotations outside cluster
  ##########################################################
  
  noncluster_SPE <- background_SPE %>%
    filter(
      !feature %in%
        cluster_features
    )
  
  ##########################################################
  # Skip empty comparisons
  ##########################################################
  
  if(
    nrow(
      cluster_SPE
    ) == 0 ||
    nrow(
      noncluster_SPE
    ) == 0
  ) {
    
    return(
      NULL
    )
  }
  
  ##########################################################
  # Taxonomy terms observed in cluster
  ##########################################################
  
  taxonomy_terms <- sort(
    unique(
      cluster_SPE$taxonomy_category
    )
  )
  
  taxonomy_terms <- taxonomy_terms[
    !is.na(
      taxonomy_terms
    ) &
      taxonomy_terms != ""
  ]
  
  if(
    length(
      taxonomy_terms
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
    taxonomy_terms,
    function(term) {
      
      ######################################################
      # Cluster counts
      ######################################################
      
      cluster_fn <- sum(
        cluster_SPE$taxonomy_category ==
          term
      )
      
      cluster_notfn <-
        nrow(
          cluster_SPE
        ) -
        cluster_fn
      
      ######################################################
      # Non-cluster counts
      ######################################################
      
      noncluster_fn <- sum(
        noncluster_SPE$taxonomy_category ==
          term
      )
      
      noncluster_notfn <-
        nrow(
          noncluster_SPE
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
      # Cluster proportion
      ######################################################
      
      cluster_proportion <-
        cluster_fn /
        nrow(
          cluster_SPE
        )
      
      ######################################################
      # Background
      ######################################################
      
      background_count <-
        cluster_fn +
        noncluster_fn
      
      background_total <-
        nrow(
          cluster_SPE
        ) +
        nrow(
          noncluster_SPE
        )
      
      background_proportion <-
        background_count /
        background_total
      
      ######################################################
      # Fold enrichment
      ######################################################
      
      fold_enrichment <- ifelse(
        background_proportion > 0,
        
        cluster_proportion /
          background_proportion,
        
        NA_real_
      )
      
      ######################################################
      # Return result
      ######################################################
      
      data.frame(
        taxonomy_level =
          taxonomy_level,
        
        taxonomy_category =
          term,
        
        cluster_count =
          cluster_fn,
        
        cluster_total =
          nrow(
            cluster_SPE
          ),
        
        noncluster_count =
          noncluster_fn,
        
        noncluster_total =
          nrow(
            noncluster_SPE
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
  # Combine enrichment results
  ##########################################################
  
  enrich_df <- bind_rows(
    enrichment_results
  )
  
  ##########################################################
  # Remove uninformative taxonomy labels
  ##########################################################
  
  enrich_df <- enrich_df %>%
    filter(
      !grepl(
        "unknown",
        taxonomy_category,
        ignore.case = TRUE
      ),
      
      !grepl(
        "unclassified",
        taxonomy_category,
        ignore.case = TRUE
      ),
      
      !grepl(
        "unassigned",
        taxonomy_category,
        ignore.case = TRUE
      )
    )
  
  ##########################################################
  # Require at least four species in cluster category
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
  # FDR correction after count filtering
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
# Run thresholds, Louvain clustering, and taxonomy enrichment
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
    # Remove self-similarity
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
    # Threshold
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
    # Square retained similarity
    ########################################################
    
    sim_mat <- sim_mat^2
    
    ########################################################
    # Retained edge list
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
  # Louvain clustering and graph summary
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
    
    g <- graph_from_data_frame(
      d = edges,
      directed = FALSE
    )
    
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
        
        modularity =
          modularity(
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
          ),
        
        stringsAsFactors = FALSE
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
  # Save threshold summary
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
  # Species enrichment for every cluster and taxonomy level
  #
  # Runs:
  # family
  # genus
  # class
  # origin
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
          cluster ==
            clust
        ) %>%
        pull(
          feature
        ) %>%
        as.character()
      
      for(
        taxonomy_level in names(
          taxonomy_levels
        )
      ) {
        
        enrich_df <- run_species_enrichment(
          cluster_features =
            cluster_features,
          
          background_features =
            background_features,
          
          taxonomy_level =
            taxonomy_level
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
              
              .before = 1
            ) %>%
            select(
              threshold,
              network,
              cluster,
              n_features_in_cluster,
              taxonomy_level,
              taxonomy_category,
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
            taxonomy_level,
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
      "louvain_threshold_summary_species.csv"
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
  
  write.csv(
    all_louvain_memberships,
    file.path(
      out_dir,
      "all_threshold_network_louvain_memberships_species.csv"
    ),
    row.names = FALSE
  )
}

############################################################
# Combine species taxonomy enrichment results
############################################################

if(
  length(
    all_enrichment_results
  ) > 0
) {
  
  all_species_enrichment_results <- bind_rows(
    all_enrichment_results
  )
  
  rownames(
    all_species_enrichment_results
  ) <- NULL
  
  write.csv(
    all_species_enrichment_results,
    file.path(
      out_dir,
      "all_threshold_cluster_species_enrichment_results.csv"
    ),
    row.names = FALSE
  )
  
  ##########################################################
  # Write separate files for each taxonomy level
  #
  # Automatically includes origin.
  ##########################################################
  
  for(
    current_taxonomy_level in names(
      taxonomy_levels
    )
  ) {
    
    level_results <- all_species_enrichment_results %>%
      filter(
        taxonomy_level ==
          current_taxonomy_level
      )
    
    write.csv(
      level_results,
      file.path(
        out_dir,
        paste0(
          "all_threshold_cluster_species_",
          current_taxonomy_level,
          "_enrichment_results.csv"
        )
      ),
      row.names = FALSE
    )
  }
  
  ##########################################################
  # Top enrichment per threshold/network/cluster/level
  ##########################################################
  
  top_species_enrichments <-
    all_species_enrichment_results %>%
    group_by(
      threshold,
      network,
      cluster,
      taxonomy_level
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
    top_species_enrichments,
    file.path(
      out_dir,
      paste0(
        "top_species_enrichments_by_threshold_",
        "network_cluster_level.csv"
      )
    ),
    row.names = FALSE
  )
  
  print(
    top_species_enrichments
  )
  
  ##########################################################
  # Significant taxonomy enrichment results
  ##########################################################
  
  significant_species_enrichments <-
    all_species_enrichment_results %>%
    filter(
      FDR <
        0.05
    ) %>%
    arrange(
      threshold,
      network,
      cluster,
      taxonomy_level,
      FDR,
      p_value
    )
  
  write.csv(
    significant_species_enrichments,
    file.path(
      out_dir,
      paste0(
        "significant_threshold_cluster_",
        "species_enrichments_FDR_0.05.csv"
      )
    ),
    row.names = FALSE
  )
  
} else {
  
  warning(
    paste0(
      "No species taxonomy enrichment results passed the ",
      "minimum cluster count of four."
    )
  )
}