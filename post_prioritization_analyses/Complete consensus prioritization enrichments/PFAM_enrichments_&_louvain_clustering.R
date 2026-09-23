library(dplyr)
library(readr)
library(tidyr)
library(igraph)
library(stringr)

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

thresholds <- c(0.50,0.60,0.70, 0.80, 0.90, 0.95, 0.975)

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
      file.path(class_dir, "avg_sig_type_mat_meat.csv"),
      row.names = 1,
      check.names = FALSE
    )
  ),
  dairy = as.matrix(
    read.csv(
      file.path(class_dir, "avg_sig_type_mat_dairy.csv"),
      row.names = 1,
      check.names = FALSE
    )
  ),
  base = as.matrix(
    read.csv(
      file.path(class_dir, "avg_sig_type_mat_base.csv"),
      row.names = 1,
      check.names = FALSE
    )
  ),
  end = as.matrix(
    read.csv(
      file.path(class_dir, "avg_sig_type_mat_end.csv"),
      row.names = 1,
      check.names = FALSE
    )
  )
)

############################################################
# Pull in names used in classification analysis
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
pecan_matrix_128$nodes <- rownames(pecan_matrix_128)
pecan_matrix_128 <- pecan_matrix_128[, c(129, 1:128)]

used_features <- pecan_matrix_128$nodes

############################################################
# Read and prepare PFAM annotations
############################################################

PFAM_pre <- as.data.frame(
  read_tsv(
    paste0(
      "C:/Users/priceade/Desktop/Fecal_Omics/Older Code and Workflows/",
      "PROTDIETsqm140V001.PFAM.lookup.tsv"
    ),
    show_col_types = FALSE
  )
)

colnames(PFAM_pre) <- c(
  "feature",
  "name"
)

PFAMgo <- read.csv(
  paste0(
    "C:/Users/priceade/Desktop/Fecal_Omics/Older Code and Workflows/",
    "pfam_sep.csv"
  ),
  header = FALSE,
  stringsAsFactors = FALSE
)

colnames(PFAMgo) <- c(
  "features",
  "details",
  "gonum"
)

############################################################
# Extract PFAM feature ID from the features column
############################################################

PFAMgo[c("feature", "p")] <- str_split_fixed(
  PFAMgo$features,
  " ",
  2
)

############################################################
# Merge PFAM annotation information
############################################################

full_PFAM <- merge(
  PFAMgo,
  PFAM_pre,
  by = "feature"
)

PFAM <- full_PFAM %>%
  select(
    feature,
    details
  ) %>%
  filter(
    !is.na(feature),
    feature != "",
    !is.na(details),
    details != ""
  )

############################################################
# Filter to PFAM features used in classification analysis
############################################################

used_PFAM <- PFAM %>%
  filter(feature %in% used_features)

############################################################
# Function for PFAM enrichment for one cluster
############################################################

run_PFAM_enrichment <- function(
    cluster_features,
    background_features
) {
  
  ##########################################################
  # PFAM annotations inside the cluster
  ##########################################################
  
  cluster_PFAM <- used_PFAM %>%
    filter(feature %in% cluster_features)
  
  ##########################################################
  # PFAM annotations in the full network background
  ##########################################################
  
  background_PFAM <- used_PFAM %>%
    filter(feature %in% background_features)
  
  ##########################################################
  # PFAM annotations outside the cluster
  ##########################################################
  
  noncluster_PFAM <- background_PFAM %>%
    filter(!feature %in% cluster_features)
  
  if(
    nrow(cluster_PFAM) == 0 ||
    nrow(noncluster_PFAM) == 0
  ) {
    return(NULL)
  }
  
  ##########################################################
  # Test PFAM terms observed in the cluster
  ##########################################################
  
  PFAM_terms <- sort(
    unique(cluster_PFAM$details)
  )
  
  PFAM_terms <- PFAM_terms[
    !is.na(PFAM_terms) &
      PFAM_terms != ""
  ]
  
  if(length(PFAM_terms) == 0) {
    return(NULL)
  }
  
  enrichment_results <- lapply(
    PFAM_terms,
    function(term) {
      
      cluster_fn <- sum(
        cluster_PFAM$details == term
      )
      
      cluster_notfn <- nrow(cluster_PFAM) -
        cluster_fn
      
      noncluster_fn <- sum(
        noncluster_PFAM$details == term
      )
      
      noncluster_notfn <- nrow(noncluster_PFAM) -
        noncluster_fn
      
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
      
      cluster_proportion <- cluster_fn /
        nrow(cluster_PFAM)
      
      background_count <- cluster_fn +
        noncluster_fn
      
      background_total <- nrow(cluster_PFAM) +
        nrow(noncluster_PFAM)
      
      background_proportion <- background_count /
        background_total
      
      fold_enrichment <- ifelse(
        background_proportion > 0,
        cluster_proportion / background_proportion,
        NA_real_
      )
      
      data.frame(
        PFAM_category = term,
        cluster_count = cluster_fn,
        cluster_total = nrow(cluster_PFAM),
        noncluster_count = noncluster_fn,
        noncluster_total = nrow(noncluster_PFAM),
        background_count = background_count,
        background_total = background_total,
        cluster_proportion = cluster_proportion,
        background_proportion = background_proportion,
        fold_enrichment = fold_enrichment,
        p_value = test$p.value
      )
    }
  )
  
  enrich_df <- bind_rows(
    enrichment_results
  )
  
  ##########################################################
  # Remove uninformative PFAM terms
  ##########################################################
  
  enrich_df <- enrich_df %>%
    filter(
      !grepl(
        "Function unknown",
        PFAM_category,
        ignore.case = TRUE
      )
    )
  
  ##########################################################
  # Require at least four occurrences in the cluster
  ##########################################################
  
  enrich_df <- enrich_df %>%
    filter(cluster_count >= 4)
  
  if(nrow(enrich_df) == 0) {
    return(NULL)
  }
  
  ##########################################################
  # FDR correction after count filtering
  ##########################################################
  
  enrich_df <- enrich_df %>%
    mutate(
      FDR = p.adjust(
        p_value,
        method = "fdr",
        n = length(p_value)
      )
    ) %>%
    arrange(
      FDR,
      p_value,
      desc(fold_enrichment)
    )
  
  return(enrich_df)
}

############################################################
# Run thresholds, Louvain clustering, and enrichment
############################################################

all_louvain_summaries <- list()
all_enrichment_results <- list()
all_membership_results <- list()

set.seed(10)

for(thresh in thresholds) {
  
  message(
    "Running threshold: ",
    thresh
  )
  
  edge_list_results <- list()
  louvain_results <- list()
  louvain_memberships <- list()
  graph_results <- list()
  
  ##########################################################
  # Create edge lists for this threshold
  ##########################################################
  
  for(type in names(avg_mats)) {
    
    sim_mat <- avg_mats[[type]]
    
    if(
      is.null(rownames(sim_mat)) ||
      is.null(colnames(sim_mat))
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
        rownames(sim_mat),
        colnames(sim_mat)
      )
    ) {
      
      stop(
        paste0(
          "Row and column names do not match for network: ",
          type
        )
      )
    }
    
    diag(sim_mat) <- 0
    
    edges_tmp <- data.frame(
      Source = rownames(sim_mat)[
        row(sim_mat)[upper.tri(sim_mat)]
      ],
      Target = colnames(sim_mat)[
        col(sim_mat)[upper.tri(sim_mat)]
      ],
      width = sim_mat[
        upper.tri(sim_mat)
      ]
    )
    
    edges_tmp <- edges_tmp %>%
      filter(is.finite(width))
    
    if(nrow(edges_tmp) == 0) {
      
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
    
    threshold_value <- quantile(
      edges_tmp$width,
      probs = thresh,
      na.rm = TRUE,
      names = FALSE
    )
    
    sim_mat[
      is.na(sim_mat) |
        sim_mat < threshold_value
    ] <- 0
    
    sim_mat <- sim_mat^2
    
    edges <- data.frame(
      Source = rownames(sim_mat)[
        row(sim_mat)[upper.tri(sim_mat)]
      ],
      Target = colnames(sim_mat)[
        col(sim_mat)[upper.tri(sim_mat)]
      ],
      width = sim_mat[
        upper.tri(sim_mat)
      ]
    )
    
    edges <- edges %>%
      filter(
        is.finite(width),
        width > 0
      )
    
    edge_list_results[[type]] <- edges
  }
  
  ##########################################################
  # Louvain clustering for this threshold
  ##########################################################
  
  for(type in names(edge_list_results)) {
    
    edges <- edge_list_results[[type]]
    
    if(nrow(edges) == 0) {
      
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
    
    # Save the graph separately because the Louvain object does not
    # reliably retain the original graph as fit$graph
    graph_results[[type]] <- g
    
    louvain_results[[type]] <- louvain_fit
    
    membership_df <- data.frame(
      feature = names(
        membership(louvain_fit)
      ),
      cluster = as.numeric(
        membership(louvain_fit)
      ),
      network = type,
      threshold = thresh
    )
    
    louvain_memberships[[type]] <- membership_df
    
    all_membership_results[[
      paste(
        thresh,
        type,
        sep = "_"
      )
    ]] <- membership_df
  }
  
  ##########################################################
  # Summary for this threshold
  ##########################################################
  
  if(length(louvain_results) > 0) {
    
    threshold_summary <- bind_rows(
      lapply(
        names(louvain_results),
        function(type) {
          
          fit <- louvain_results[[type]]
          graph_object <- graph_results[[type]]
          memb <- membership(fit)
          cluster_sizes <- table(memb)
          
          data.frame(
            threshold = thresh,
            network = type,
            n_nodes = vcount(graph_object),
            n_edges = ecount(graph_object),
            n_clusters = length(cluster_sizes),
            modularity = modularity(
              graph_object,
              membership = memb,
              weights = E(graph_object)$width
            ),
            largest_cluster = as.numeric(
              max(cluster_sizes)
            ),
            smallest_cluster = as.numeric(
              min(cluster_sizes)
            ),
            mean_cluster_size = round(
              mean(
                as.numeric(cluster_sizes)
              ),
              2
            ),
            median_cluster_size = round(
              median(
                as.numeric(cluster_sizes)
              ),
              2
            )
          )
        }
      )
    )
    
    all_louvain_summaries[[
      as.character(thresh)
    ]] <- threshold_summary
  }
  
  ##########################################################
  # PFAM enrichment for every cluster
  ##########################################################
  
  for(type in names(louvain_memberships)) {
    
    membership_df <- louvain_memberships[[type]]
    
    background_features <- membership_df$feature
    
    for(clust in sort(
      unique(membership_df$cluster)
    )) {
      
      cluster_features <- membership_df %>%
        filter(cluster == clust) %>%
        pull(feature)
      
      enrich_df <- run_PFAM_enrichment(
        cluster_features = cluster_features,
        background_features = background_features
      )
      
      if(
        !is.null(enrich_df) &&
        nrow(enrich_df) > 0
      ) {
        
        enrich_df <- enrich_df %>%
          mutate(
            threshold = thresh,
            network = type,
            cluster = clust,
            n_features_in_cluster = length(
              cluster_features
            ),
            .before = 1
          ) %>%
          select(
            threshold,
            network,
            cluster,
            n_features_in_cluster,
            PFAM_category,
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
          sep = "_"
        )
        
        all_enrichment_results[[
          result_name
        ]] <- enrich_df
      }
    }
  }
}

############################################################
# Combine Louvain summaries
############################################################

if(length(all_louvain_summaries) > 0) {
  
  louvain_threshold_summary <- bind_rows(
    all_louvain_summaries
  )
  
  rownames(louvain_threshold_summary) <- NULL
  
  write.csv(
    louvain_threshold_summary,
    file.path(
      out_dir,
      "louvain_threshold_summary_PFAM.csv"
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

if(length(all_membership_results) > 0) {
  
  all_louvain_memberships <- bind_rows(
    all_membership_results
  )
  
  rownames(all_louvain_memberships) <- NULL
  
  write.csv(
    all_louvain_memberships,
    file.path(
      out_dir,
      "all_threshold_network_louvain_memberships_PFAM.csv"
    ),
    row.names = FALSE
  )
}

############################################################
# Combine PFAM enrichment results
############################################################

if(length(all_enrichment_results) > 0) {
  
  all_PFAM_enrichment_results <- bind_rows(
    all_enrichment_results
  )
  
  rownames(all_PFAM_enrichment_results) <- NULL
  
  write.csv(
    all_PFAM_enrichment_results,
    file.path(
      out_dir,
      "all_threshold_cluster_PFAM_enrichment_results.csv"
    ),
    row.names = FALSE
  )
  
  ##########################################################
  # Top enrichment per threshold/network/cluster
  ##########################################################
  
  top_PFAM_enrichments <- all_PFAM_enrichment_results %>%
    group_by(
      threshold,
      network,
      cluster
    ) %>%
    arrange(
      FDR,
      p_value,
      desc(fold_enrichment),
      .by_group = TRUE
    ) %>%
    slice_head(n = 5) %>%
    ungroup()
  
  write.csv(
    top_PFAM_enrichments,
    file.path(
      out_dir,
      paste0(
        "top_PFAM_enrichments_by_threshold_",
        "network_cluster.csv"
      )
    ),
    row.names = FALSE
  )
  
  print(
    top_PFAM_enrichments
  )
  
  ##########################################################
  # Significant PFAM enrichment results
  ##########################################################
  
  significant_PFAM_enrichments <- all_PFAM_enrichment_results %>%
    filter(FDR < 0.05) %>%
    arrange(
      threshold,
      network,
      cluster,
      FDR,
      p_value
    )
  
  write.csv(
    significant_PFAM_enrichments,
    file.path(
      out_dir,
      paste0(
        "significant_threshold_cluster_",
        "PFAM_enrichments_FDR_0.05.csv"
      )
    ),
    row.names = FALSE
  )
  
} else {
  
  warning(
    paste0(
      "No PFAM enrichment results passed the ",
      "minimum cluster count of four."
    )
  )
}