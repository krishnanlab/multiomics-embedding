library(readr)
library(readxl)
#library(janitor)
library(dplyr)
library(tibble)
library(tidyverse)


#read in background data and raw data
db_COG <- read_tsv("C:/Users/priceade/Desktop/Fecal_Omics/Older Code and Workflows/original_data/PROTDIETsqm140V001.COG.abund_v01.tsv")
db_KO <-read_tsv("C:/Users/priceade/Desktop/Fecal_Omics/Older Code and Workflows/original_data/PROTDIETsqm140V001.KO.abund_v01.tsv")
db_PFAM <- read.table("C:/Users/priceade/Desktop/Fecal_Omics/Older Code and Workflows/original_data/PROTDIETsqm140V001.PFAM.abund_v02.txt")
this <- read.table("C:/Users/priceade/Desktop/Fecal_Omics/Older Code and Workflows/original_data/PROTDIETsqm140V001.PFAM.abund_v02.txt", header = TRUE)
col_X <- colnames(this)
Col <- sub("X","",col_X)
colnames(this)<-Col
db_PFAM <- this
db_species <- read_tsv("C:/Users/priceade/Desktop/Fecal_Omics/Older Code and Workflows/original_data/PROTDIETsqm140V001.species.allfilter.abund_v01.tsv")
metadata <- read_excel("C:/Users/priceade/Desktop/Fecal_Omics/Older Code and Workflows/original_data/PROTDIET_metadata_21Sept2022.xls")
Library_to_remove <- metadata[metadata$Include250K == "N",]$Library
info_data <- metadata[,c("Library","ID","Timepoint","WAZ","WLZ","LAZ","HCZ", "Group")]
info_data$Time <- ifelse(info_data$Timepoint == "BL", "Base", "End")




#COG
file.name<-"COG"
t_COG <- data.frame(t(db_COG))
colnames(t_COG) <-t_COG[1,]
t_COG <- t_COG[-1,]
t_COG <- rownames_to_column(t_COG,"Library")
dim(t_COG)
t_COG <-t_COG[!t_COG$Library %in% Library_to_remove,]
dim(t_COG)
COG_cpds <- colnames(t_COG)[2:ncol(t_COG)]
rownames(t_COG) <- t_COG$Library

dim(t_COG) #22656 compounds
total_lib <- nrow(t_COG)


#turn into numeric
t_COG_num <- apply(t_COG[,COG_cpds], 2, as.numeric)
rownames(t_COG_num)<-t_COG$Library
t_COG_data <- cbind(t_COG$Library,t_COG_num)
colnames(t_COG_data)[1] <- "Library"
missing_values_counts <- apply(t_COG_num, 2, function(x) sum(x == 0))
missing_values_pcts <- apply(t_COG_num,2, function(x) sum(x==0)/total_lib)

missing_df <- data.frame(feature = names(missing_values_counts),
                         Missing_Count = missing_values_counts,
                         Missing_Pct = missing_values_pcts)


############remove no features based on initial missingness############
COG_cpds_2 <- missing_df$feature[missing_df$Missing_Pct<2]
t_COG2 <- t_COG_num[,COG_cpds_2]
dim(t_COG2)
total_row_COG <- rowSums(t_COG2)
total_matrix_COG <- matrix(nrow= nrow(t_COG2),rep(total_row_COG,length(COG_cpds_2)))
colnames(total_matrix_COG) <-COG_cpds_2 
rownames(total_matrix_COG) <- rownames(t_COG2)
total_matrix_COG_2 <- t_COG2/total_matrix_COG*100
dim(total_matrix_COG_2)


#remove any features that have less than .01% abundance, per sample, across all observational units
t_COG3 <- total_matrix_COG_2[, apply(total_matrix_COG_2, 2, function(X) any(abs(X)>0.01))]
dim(t_COG3)

#coeff  of var filtering (none here)
feature <- colnames(t_COG3)
t_COG_mean <- colMeans(t_COG3)
t_COG_sd <- sapply(data.frame(t_COG3),sd)
mean <-t_COG_mean  
sd <- t_COG_sd
cv_data <- data.frame(feature, mean, sd)
cv_data$cv <- cv_data$mean/cv_data$sd
filter_df <- inner_join(cv_data,missing_df)
t_COG4 <- filter_df
dim(t_COG4)
COG_features <- t_COG4$feature
t_COG4_correct <- filter_df
dim(t_COG4_correct)
COG_features_correct <- t_COG4_correct$feature


#add info back to columns
info_data <- metadata[,c("Library","ID","Timepoint","WAZ","WLZ","LAZ","HCZ", "Group")]
info_data$Time <- ifelse(info_data$Timepoint == "BL", "Base", "End")
colnames(metadata)

final_tCOG <- data.frame(t_COG3[,COG_features])
dim(final_tCOG)

Library<-rownames(final_tCOG)

final_tCOG["Library"] = Library
final_tCOG <- final_tCOG[,c(8564,1:8563)]

######################################Calculate missingness dist##################################
pre_missing_COG <- as.matrix(final_tCOG[,2:8564])

missing_values_counts <- apply(pre_missing_COG, 2, function(x) sum(x == 0))
missing_values_pcts <- apply(pre_missing_COG, 2, function(x) sum(x==0)/total_lib)

final_tCOG_missingness <- data.frame(feature = names(missing_values_counts),
                         Missing_Count = missing_values_counts,
                         Missing_Pct = missing_values_pcts)





#KO
file.name<-"KO"
t_KO <- data.frame(t(db_KO))
colnames(t_KO) <-t_KO[1,]
t_KO <- t_KO[-1,]
t_KO <- rownames_to_column(t_KO,"Library")
dim(t_KO)
t_KO <-t_KO[!t_KO$Library %in% Library_to_remove,]
dim(t_KO)
KO_cpds <- colnames(t_KO)[2:ncol(t_KO)]
rownames(t_KO) <- t_KO$Library

dim(t_KO) #22656 compounds
total_lib <- nrow(t_KO)


#turn into numeric
t_KO_num <- apply(t_KO[,KO_cpds], 2, as.numeric)
rownames(t_KO_num)<-t_KO$Library
t_KO_data <- cbind(t_KO$Library,t_KO_num)
colnames(t_KO_data)[1] <- "Library"
missing_values_counts <- apply(t_KO_num, 2, function(x) sum(x == 0))
missing_values_pcts <- apply(t_KO_num,2, function(x) sum(x==0)/total_lib)

missing_df <- data.frame(feature = names(missing_values_counts),
                         Missing_Count = missing_values_counts,
                         Missing_Pct = missing_values_pcts)


############remove no features based on initial missingness############
KO_cpds_2 <- missing_df$feature[missing_df$Missing_Pct<2]
t_KO2 <- t_KO_num[,KO_cpds_2]
dim(t_KO2)
total_row_KO <- rowSums(t_KO2)
total_matrix_KO <- matrix(nrow= nrow(t_KO2),rep(total_row_KO,length(KO_cpds_2)))
colnames(total_matrix_KO) <-KO_cpds_2 
rownames(total_matrix_KO) <- rownames(t_KO2)
total_matrix_KO_2 <- t_KO2/total_matrix_KO*100
dim(total_matrix_KO_2)


#remove any features that have less than .01% abundance, per sample, across all observational units
t_KO3 <- total_matrix_KO_2[, apply(total_matrix_KO_2, 2, function(X) any(abs(X)>0.01))]
dim(t_KO3)

#coeff  of var filtering (none here)
feature <- colnames(t_KO3)
t_KO_mean <- colMeans(t_KO3)
t_KO_sd <- sapply(data.frame(t_KO3),sd)
mean <-t_KO_mean  
sd <- t_KO_sd
cv_data <- data.frame(feature, mean, sd)
cv_data$cv <- cv_data$mean/cv_data$sd
filter_df <- inner_join(cv_data,missing_df)
t_KO4 <- filter_df
dim(t_KO4)
KO_features <- t_KO4$feature
t_KO4_correct <- filter_df
dim(t_KO4_correct)
KO_features_correct <- t_KO4_correct$feature


#add info back to columns
info_data <- metadata[,c("Library","ID","Timepoint","WAZ","WLZ","LAZ","HCZ", "Group")]
info_data$Time <- ifelse(info_data$Timepoint == "BL", "Base", "End")
colnames(metadata)

final_tKO <- data.frame(t_KO3[,KO_features])
dim(final_tKO)

Library<-rownames(final_tKO)

final_tKO["Library"] = Library
final_tKO <- final_tKO[,c(4434,1:4433)]

######################################Calculate missingness dist##################################
pre_missing_KO <- as.matrix(final_tKO[,2:4434])

missing_values_counts <- apply(pre_missing_KO, 2, function(x) sum(x == 0))
missing_values_pcts <- apply(pre_missing_KO, 2, function(x) sum(x==0)/total_lib)

final_tKO_missingness <- data.frame(feature = names(missing_values_counts),
                                     Missing_Count = missing_values_counts,
                                     Missing_Pct = missing_values_pcts)






#PFAM
file.name<-"PFAM"
t_PFAM <- data.frame(t(db_PFAM))
colnames(t_PFAM) <-t_PFAM[1,]
t_PFAM <- t_PFAM[-1,]
t_PFAM <- rownames_to_column(t_PFAM,"Library")
dim(t_PFAM)
t_PFAM <-t_PFAM[!t_PFAM$Library %in% Library_to_remove,]
dim(t_PFAM)
PFAM_cpds <- colnames(t_PFAM)[2:ncol(t_PFAM)]
rownames(t_PFAM) <- t_PFAM$Library

dim(t_PFAM) #22656 compounds
total_lib <- nrow(t_PFAM)


#turn into numeric
t_PFAM_num <- apply(t_PFAM[,PFAM_cpds], 2, as.numeric)
rownames(t_PFAM_num)<-t_PFAM$Library
t_PFAM_data <- cbind(t_PFAM$Library,t_PFAM_num)
colnames(t_PFAM_data)[1] <- "Library"
missing_values_counts <- apply(t_PFAM_num, 2, function(x) sum(x == 0))
missing_values_pcts <- apply(t_PFAM_num,2, function(x) sum(x==0)/total_lib)

missing_df <- data.frame(feature = names(missing_values_counts),
                         Missing_Count = missing_values_counts,
                         Missing_Pct = missing_values_pcts)


############remove no features based on initial missingness############
PFAM_cpds_2 <- missing_df$feature[missing_df$Missing_Pct<2]
t_PFAM2 <- t_PFAM_num[,PFAM_cpds_2]
dim(t_PFAM2)
total_row_PFAM <- rowSums(t_PFAM2)
total_matrix_PFAM <- matrix(nrow= nrow(t_PFAM2),rep(total_row_PFAM,length(PFAM_cpds_2)))
colnames(total_matrix_PFAM) <-PFAM_cpds_2 
rownames(total_matrix_PFAM) <- rownames(t_PFAM2)
total_matrix_PFAM_2 <- t_PFAM2/total_matrix_PFAM*100
dim(total_matrix_PFAM_2)


#remove any features that have less than .01% abundance, per sample, across all observational units
t_PFAM3 <- total_matrix_PFAM_2[, apply(total_matrix_PFAM_2, 2, function(X) any(abs(X)>0.01))]
dim(t_PFAM3)

#coeff  of var filtering (none here)
feature <- colnames(t_PFAM3)
t_PFAM_mean <- colMeans(t_PFAM3)
t_PFAM_sd <- sapply(data.frame(t_PFAM3),sd)
mean <-t_PFAM_mean  
sd <- t_PFAM_sd
cv_data <- data.frame(feature, mean, sd)
cv_data$cv <- cv_data$mean/cv_data$sd
filter_df <- inner_join(cv_data,missing_df)
t_PFAM4 <- filter_df
dim(t_PFAM4)
PFAM_features <- t_PFAM4$feature
t_PFAM4_correct <- filter_df
dim(t_PFAM4_correct)
PFAM_features_correct <- t_PFAM4_correct$feature


#add info back to columns
info_data <- metadata[,c("Library","ID","Timepoint","WAZ","WLZ","LAZ","HCZ", "Group")]
info_data$Time <- ifelse(info_data$Timepoint == "BL", "Base", "End")
colnames(metadata)

final_tPFAM <- data.frame(t_PFAM3[,PFAM_features])
dim(final_tPFAM)

Library<-rownames(final_tPFAM)

final_tPFAM["Library"] = Library
final_tPFAM <- final_tPFAM[,c(3702,1:3701)]

######################################Calculate missingness dist##################################
pre_missing_PFAM <- as.matrix(final_tPFAM[,2:3702])

missing_values_counts <- apply(pre_missing_PFAM, 2, function(x) sum(x == 0))
missing_values_pcts <- apply(pre_missing_PFAM, 2, function(x) sum(x==0)/total_lib)

final_tPFAM_missingness <- data.frame(feature = names(missing_values_counts),
                                    Missing_Count = missing_values_counts,
                                    Missing_Pct = missing_values_pcts)



#species
file.name<-"species"
t_species <- data.frame(t(db_species))
colnames(t_species) <-t_species[1,]
t_species <- t_species[-1,]
t_species <- rownames_to_column(t_species,"Library")
dim(t_species)
t_species <-t_species[!t_species$Library %in% Library_to_remove,]
dim(t_species)
species_cpds <- colnames(t_species)[2:ncol(t_species)]
rownames(t_species) <- t_species$Library

dim(t_species) #22656 compounds
total_lib <- nrow(t_species)


#turn into numeric
t_species_num <- apply(t_species[,species_cpds], 2, as.numeric)
rownames(t_species_num)<-t_species$Library
t_species_data <- cbind(t_species$Library,t_species_num)
colnames(t_species_data)[1] <- "Library"
missing_values_counts <- apply(t_species_num, 2, function(x) sum(x == 0))
missing_values_pcts <- apply(t_species_num,2, function(x) sum(x==0)/total_lib)

missing_df <- data.frame(feature = names(missing_values_counts),
                         Missing_Count = missing_values_counts,
                         Missing_Pct = missing_values_pcts)


############remove no features based on initial missingness############
species_cpds_2 <- missing_df$feature[missing_df$Missing_Pct<2]
t_species2 <- t_species_num[,species_cpds_2]
dim(t_species2)
total_row_species <- rowSums(t_species2)
total_matrix_species <- matrix(nrow= nrow(t_species2),rep(total_row_species,length(species_cpds_2)))
colnames(total_matrix_species) <-species_cpds_2 
rownames(total_matrix_species) <- rownames(t_species2)
total_matrix_species_2 <- t_species2/total_matrix_species*100
dim(total_matrix_species_2)


#remove any features that have less than .01% abundance, per sample, across all observational units
t_species3 <- total_matrix_species_2[, apply(total_matrix_species_2, 2, function(X) any(abs(X)>0.01))]
dim(t_species3)

#coeff  of var filtering (none here)
feature <- colnames(t_species3)
t_species_mean <- colMeans(t_species3)
t_species_sd <- sapply(data.frame(t_species3),sd)
mean <-t_species_mean  
sd <- t_species_sd
cv_data <- data.frame(feature, mean, sd)
cv_data$cv <- cv_data$mean/cv_data$sd
filter_df <- inner_join(cv_data,missing_df)
t_species4 <- filter_df
dim(t_species4)
species_features <- t_species4$feature
t_species4_correct <- filter_df
dim(t_species4_correct)
species_features_correct <- t_species4_correct$feature


#add info back to columns
info_data <- metadata[,c("Library","ID","Timepoint","WAZ","WLZ","LAZ","HCZ", "Group")]
info_data$Time <- ifelse(info_data$Timepoint == "BL", "Base", "End")
colnames(metadata)

final_tspecies <- data.frame(t_species3[,species_features])
dim(final_tspecies)

Library<-rownames(final_tspecies)

final_tspecies["Library"] = Library
final_tspecies <- final_tspecies[,c(477,1:476)]

######################################Calculate missingness dist##################################
pre_missing_species <- as.matrix(final_tspecies[,2:477])

missing_values_counts <- apply(pre_missing_species, 2, function(x) sum(x == 0))
missing_values_pcts <- apply(pre_missing_species, 2, function(x) sum(x==0)/total_lib)

final_tspecies_missingness <- data.frame(feature = names(missing_values_counts),
                                    Missing_Count = missing_values_counts,
                                    Missing_Pct = missing_values_pcts)



pre_missing_COG_an <- pre_missing_COG
pre_missing_KO_an <- pre_missing_KO
pre_missing_PFAM_an <- pre_missing_PFAM
pre_missing_species_an <- pre_missing_species



missing_values_counts <- apply(pre_missing_COG_an, 2, function(x) sum(x == 0))
missing_values_pcts <- apply(pre_missing_COG_an, 2, function(x) sum(x==0)/total_lib)
final_tCOG_missingness <- data.frame(feature = names(missing_values_counts),
                                         Missing_Count = missing_values_counts,
                                         Missing_Pct = missing_values_pcts)

missing_values_counts <- apply(pre_missing_KO_an, 2, function(x) sum(x == 0))
missing_values_pcts <- apply(pre_missing_KO_an, 2, function(x) sum(x==0)/total_lib)
final_tKO_missingness <- data.frame(feature = names(missing_values_counts),
                                         Missing_Count = missing_values_counts,
                                         Missing_Pct = missing_values_pcts)

missing_values_counts <- apply(pre_missing_PFAM_an, 2, function(x) sum(x == 0))
missing_values_pcts <- apply(pre_missing_PFAM_an, 2, function(x) sum(x==0)/total_lib)
final_tPFAM_missingness <- data.frame(feature = names(missing_values_counts),
                                         Missing_Count = missing_values_counts,
                                         Missing_Pct = missing_values_pcts)

missing_values_counts <- apply(pre_missing_species_an, 2, function(x) sum(x == 0))
missing_values_pcts <- apply(pre_missing_species_an, 2, function(x) sum(x==0)/total_lib)
final_tspecies_missingness <- data.frame(feature = names(missing_values_counts),
                                         Missing_Count = missing_values_counts,
                                         Missing_Pct = missing_values_pcts)


##############################Calculate CV per group#######################
feature <- colnames(pre_missing_COG_an)
mean <- colMeans(pre_missing_COG_an)
sd <- sapply(data.frame(pre_missing_COG_an),sd)
final_tCOG_missingness$cv <- mean/sd

feature <- colnames(pre_missing_PFAM_an)
mean <- colMeans(pre_missing_PFAM_an)
sd <- sapply(data.frame(pre_missing_PFAM_an),sd)
final_tPFAM_missingness$cv <- mean/sd

feature <- colnames(pre_missing_KO_an)
mean <- colMeans(pre_missing_KO_an)
sd <- sapply(data.frame(pre_missing_KO_an),sd)
final_tKO_missingness$cv <- mean/sd

feature <- colnames(pre_missing_species_an)
mean <- colMeans(pre_missing_species_an)
sd <- sapply(data.frame(pre_missing_species_an),sd)
final_tspecies_missingness$cv <- mean/sd

#####################################filter for cv or min missingness##############
cv_.02_COG <- quantile(final_tCOG_missingness$cv,.02)
cv_.02_KO <- quantile(final_tKO_missingness$cv,.02)
cv_.02_PFAM <- quantile(final_tPFAM_missingness$cv,.02)
cv_.02_species <- quantile(final_tspecies_missingness$cv,.02)


#####################################################################################
COG_missing_less90 <- final_tCOG_missingness[final_tCOG_missingness$Missing_Pct < .9,] 
PFAM_missing_less90 <- final_tPFAM_missingness[final_tPFAM_missingness$Missing_Pct < .9,] 
KO_missing_less90 <- final_tKO_missingness[final_tKO_missingness$Missing_Pct < .9,] 
species_missing_less90 <- final_tspecies_missingness[final_tspecies_missingness$Missing_Pct < .9,] 

COG_cv_miss_filt <- COG_missing_less90[(COG_missing_less90$cv > cv_.02_COG) | (COG_missing_less90$Missing_Pct > .1),]
KO_cv_miss_filt <- KO_missing_less90[(KO_missing_less90$cv > cv_.02_KO) | (KO_missing_less90$Missing_Pct > .1),]
PFAM_cv_miss_filt <- PFAM_missing_less90[(PFAM_missing_less90$cv > cv_.02_PFAM) | (PFAM_missing_less90$Missing_Pct > .1),]
species_cv_miss_filt <- species_missing_less90[(species_missing_less90$cv > cv_.02_species) | (species_missing_less90$Missing_Pct > .1),]


COG_miss_filt <- pre_missing_COG_an[,COG_cv_miss_filt$feature] 
PFAM_miss_filt <- pre_missing_PFAM_an[,PFAM_cv_miss_filt$feature]
KO_miss_filt <- pre_missing_KO_an[,KO_cv_miss_filt$feature]
species_miss_filt <- pre_missing_species_an[,species_cv_miss_filt$feature]



#testing for abundance differences#
#######################################Explore significance############################
all_micro <- cbind.data.frame('Library'=rownames(COG_miss_filt), COG_miss_filt, KO_miss_filt, PFAM_miss_filt, species_miss_filt)

pre_micro_labeled <- merge(all_micro, info_data, by.x = 'Library', by.y = 'Library')

#write.csv(pre_micro_labeled, "C:/Users/priceade/Desktop/Fecal_Omics/microbiome_data_for_differential abundance.csv", row.names = FALSE)

pre_micro_labeled <- as.data.frame(pre_micro_labeled) %>%
  mutate(across(where(is.numeric),function(x) {if_else(x==0,NA,x)}))
pre_micro_labeled <- pre_micro_labeled %>%
  mutate(across(where(is.numeric),function(x) {if_else(is.na(x),min(x,na.rm=T),x)}))


micro_labeled <- pre_micro_labeled[pre_micro_labeled$ID %in% names(which(table(pre_micro_labeled$ID) > 1)), ]

micro_base <- micro_labeled[micro_labeled$Timepoint == 'BL',]
micro_end <- micro_labeled[micro_labeled$Timepoint == 'E',]

micro_b_data <- as.matrix(micro_base[,2:17035])
class(micro_b_data) <- "numeric"
micro_e_data <- as.matrix(micro_end[,2:17035])
class(micro_e_data) <- "numeric"



#log 2 transform
micro_b_data <- log2(micro_b_data)
micro_e_data <- log2(micro_e_data)


p_val <- c()
stat <- c()

for (i in 1:dim(micro_b_data)[2]){
  test <-wilcox.test(micro_b_data[,(i)],micro_e_data[,(i)], paired = TRUE, exact = FALSE)
  p_val <- append(p_val,as.numeric(test$p.value))
  stat <- append(stat,as.numeric(test$statistic))
}

bon_cor <- .05/dim(micro_b_data)[2]
fdr_cor <- p.adjust(p_val, method = 'fdr', n= dim(micro_b_data)[2])




##########################################FOLD CHANGE CALCULATIONS########################################

#calculate the mean of each feature val per samp
base = as.vector(apply(micro_b_data, 2, mean))

#calcuate the mean of each gene per test group
end = as.vector(apply(micro_e_data, 2, mean))

fold_change <- -1*(base-end)




full_results <- cbind.data.frame('feature'=colnames(micro_b_data), 'p_val' = p_val, 'fdr' = fdr_cor, 'fold_change' = fold_change)

full_results <- na.omit(full_results)

#bonf
col_meta <- ifelse(((-1*log10(full_results$fdr) > (-1*log10(.05))) & full_results$fold_change <0), "burlywood2", 
                   ifelse(((-1*log10(full_results$fdr) > (-1*log10(.05))) & full_results$fold_change >0), "lightblue", "grey"))

col_meta <- ifelse((full_results$p_val <= .05/length(full_results$p_val) & full_results$fold_change <0), "darkorange4", 
                   ifelse((full_results$p_val <= .05/length(full_results$p_val)  & full_results$fold_change >0), "darkblue", col_meta))





###################visualize dis of sig w/ large fold change#####################
p_vals_filtered <- full_results[full_results$fdr < .05,]



#merge fold change info with key for features
key <- read.csv("C://Users/priceade/Desktop/Fecal_Omics/Older Code and Workflows/All_micro_metab_KEY.csv")

base_end_key <- merge(full_results, key, by.x = "feature", by.y = "code", all.x = TRUE)

#write.csv(base_end_key, "C://Users/priceade/Desktop/Fecal_Omics/Current_Code&Workflow/Final_Manuscript_Figures/Differential_Abundance/diff_abun_microbiome_BASE_END.csv", row.names = FALSE)





library(ggplot2)
time <- ggplot(data = full_results, aes(x = fold_change, y = -1*log10(p_val))) + 
  geom_point(colour=col_meta) + ggtitle("Microbiome: Baseline v. Endpoint") +
  geom_hline(yintercept=-1*log10(max(p_vals_filtered$p_val)), linetype='solid', col = 'darkgrey') +
  geom_hline(yintercept=-1*log10(.05/length(p_val)), linetype='solid', col = 'black') +
  xlab('log2 fold change') +ylab('-log10(p-values)') +
  theme(axis.line = element_line(colour = "black"), 
        panel.border = element_rect(color = "black", fill = NA, linewidth = 1), 
        axis.text=element_text(size=20),
        axis.title=element_text(size=20,face="bold"),
        plot.title = element_text(size = 20, face = "bold", hjust = 0.5)) 

print(time)



########################################Explore meat/dairy significance############################
pre_micro_labeled <- pre_micro_labeled[pre_micro_labeled$Time == 'End',] 
micro_dairy <- pre_micro_labeled[pre_micro_labeled$Group == 'Dairy',]
micro_meat <- pre_micro_labeled[pre_micro_labeled$Group == 'Meat',]

micro_d_data <- as.matrix(micro_dairy[,2:17035])
class(micro_d_data) <- "numeric"
micro_m_data <- as.matrix(micro_meat[,2:17035])
class(micro_m_data) <- "numeric"

micro_d_data <- log2(micro_d_data)
micro_m_data <- log2(micro_m_data)

p_val <- c()
stat <- c()

for (i in 1:dim(micro_d_data)[2]){
  test <-wilcox.test(micro_d_data[,(i)],micro_m_data[,(i)], exact = FALSE)
  p_val <- append(p_val,as.numeric(test$p.value))
  stat <- append(stat,as.numeric(test$statistic))
}

bon_cor <- .05/dim(micro_d_data)[2]
fdr_cor <- p.adjust(p_val, method = 'fdr', n= dim(micro_d_data)[2])



##########################################FOLD CHANGE CALCULATIONS########################################
#calculate the mean of each feature val per samp
dairy = as.vector(apply(micro_d_data, 2, mean))

#calcuate the mean of each gene per test group
meat = as.vector(apply(micro_m_data, 2, mean))

fold_change <- -1*(dairy-meat)




full_results <- cbind.data.frame('feature'=colnames(micro_d_data), 'p_val' = p_val, 'fdr' = fdr_cor, 'fold_change' = fold_change)

full_results <- na.omit(full_results)

#bonf
col_meta <- ifelse(((-1*log10(full_results$fdr) > (-1*log10(.05))) & full_results$fold_change <0), "burlywood2", 
                   ifelse(((-1*log10(full_results$fdr) > (-1*log10(.05))) & full_results$fold_change >0), "lightblue", "grey"))

col_meta <- ifelse((full_results$p_val <= .05/length(full_results$p_val) & full_results$fold_change <0), "darkorange4", 
                   ifelse((full_results$p_val <= .05/length(full_results$p_val)  & full_results$fold_change >0), "darkblue", col_meta))



###################visualize dis of sig w/ large fold change#####################
p_vals_filtered <- full_results[full_results$fdr > .05,]

library(ggplot2)
diet <- ggplot(data = full_results, aes(x = fold_change, y = -1*log10(p_val))) + 
  geom_point(colour=col_meta) + ggtitle("Microbiome: Dairy v. Meat") +
  geom_hline(yintercept=-1*log10(min(p_vals_filtered$p_val)), linetype='solid', col = 'darkgrey') +
  xlab('log2 fold change') +ylab('-log10(p-values)') +
  theme(axis.line = element_line(colour = "black"), 
        panel.border = element_rect(color = "black", fill = NA, linewidth = 1), 
        axis.text=element_text(size=20),
        axis.title=element_text(size=20,face="bold"),
        plot.title = element_text(size = 20, face = "bold", hjust = 0.5)) 



#merge fold change info with key for features
key <- read.csv("C://Users/priceade/Desktop/Fecal_Omics/Older Code and Workflows/All_micro_metab_KEY.csv")
meat_dairy_key <- merge(full_results, key, by.x = "feature", by.y = "code", all.x = TRUE)
#write.csv(micro_species, "C://Users/priceade/Desktop/Fecal_Omics/Current_Code&Workflow/Enrichment_Analysis/Enrichment_Analysis_foldchange/micro_spec_base_end.enrich.csv", row.names = FALSE)
#write.csv(meat_dairy_key, "C://Users/priceade/Desktop/Fecal_Omics/Current_Code&Workflow/Final_Manuscript_Figures/Differential_Abundance/diff_abun_microbiome_MEAT_DAIRY.csv", row.names = FALSE)




















#write.csv(COG_miss_filt, "C://Users/priceade/Desktop/Fecal_Omics/Current_Code&Workflow/Data/cog_filt.csv", row.names = FALSE)
#write.csv(KO_miss_filt, "C://Users/priceade/Desktop/Fecal_Omics/Current_Code&Workflow/Data/ko_filt.csv", row.names = FALSE)
#write.csv(PFAM_miss_filt, "C://Users/priceade/Desktop/Fecal_Omics/Current_Code&Workflow/Data/pfam_filt.csv", row.names = FALSE)
#write.csv(species_miss_filt, "C://Users/priceade/Desktop/Fecal_Omics/Current_Code&Workflow/Data/species_filt.csv", row.names = FALSE)



#####################put data into final full form#############
library(scales)

#COG
rank <- apply(COG_miss_filt, 2, function(x) rank(replace(x,x==0, NA), na='keep'))
rank <- apply(rank, 2, function(x) replace(x,is.na(x), 0))
COG_rank_scaled <- as.data.frame(apply(rank, 2, function(x) (x/max(x))))
COG_rank_scaled$Library <- rownames(COG_rank_scaled) 
COG_rank_scaled_labeled <- merge(COG_rank_scaled, info_data, by.x = 'Library', by.y = 'Library')
COG_rank_scaled_labeled$sample.name <- paste0(COG_rank_scaled_labeled$ID, "_", COG_rank_scaled_labeled$Time)
dim(COG_rank_scaled_labeled)
COG_rank_scaled_labeled <- COG_rank_scaled_labeled[,c(2:8477, 8485)]
long_data_COG <- COG_rank_scaled_labeled %>%
  pivot_longer(!sample.name, names_to = "feature", values_to = "Norm_value")
long_data_COG <- long_data_COG[long_data_COG$Norm_value >0,]


#KO
rank <- apply(KO_miss_filt, 2, function(x) rank(replace(x,x==0, NA), na='keep'))
rank <- apply(rank, 2, function(x) replace(x,is.na(x), 0))
KO_rank_scaled <- as.data.frame(apply(rank, 2, function(x) (x/max(x))))
KO_rank_scaled$Library <- rownames(KO_rank_scaled) 
KO_rank_scaled_labeled <- merge(KO_rank_scaled, info_data, by.x = 'Library', by.y = 'Library')
KO_rank_scaled_labeled$sample.name <- paste0(KO_rank_scaled_labeled$ID, "_", KO_rank_scaled_labeled$Time)
dim(KO_rank_scaled_labeled)
KO_rank_scaled_labeled <- KO_rank_scaled_labeled[,c(2:4417, 4425)]
long_data_KO <- KO_rank_scaled_labeled %>%
  pivot_longer(!sample.name, names_to = "feature", values_to = "Norm_value")
long_data_KO <- long_data_KO[long_data_KO$Norm_value>0,]

#PFAM
rank <- apply(PFAM_miss_filt, 2, function(x) rank(replace(x,x==0, NA), na='keep'))
rank <- apply(rank, 2, function(x) replace(x,is.na(x), 0))
PFAM_rank_scaled <- as.data.frame(apply(rank, 2, function(x) (x/max(x))))
PFAM_rank_scaled$Library <- rownames(PFAM_rank_scaled) 
PFAM_rank_scaled_labeled <- merge(PFAM_rank_scaled, info_data, by.x = 'Library', by.y = 'Library')
PFAM_rank_scaled_labeled$sample.name <- paste0(PFAM_rank_scaled_labeled$ID, "_", PFAM_rank_scaled_labeled$Time)
dim(PFAM_rank_scaled_labeled)
PFAM_rank_scaled_labeled <- PFAM_rank_scaled_labeled[,c(3:3686, 3694)]
long_data_PFAM <- PFAM_rank_scaled_labeled %>%
  pivot_longer(!sample.name, names_to = "feature", values_to = "Norm_value")
long_data_PFAM <- long_data_PFAM[long_data_PFAM$Norm_value >0,]

#species
rank <- apply(species_miss_filt, 2, function(x) rank(replace(x,x==0, NA), na='keep'))
rank <- apply(rank, 2, function(x) replace(x,is.na(x), 0))
species_rank_scaled <- as.data.frame(apply(rank, 2, function(x) (x/max(x))))
species_rank_scaled$Library <- rownames(species_rank_scaled) 
species_rank_scaled_labeled <- merge(species_rank_scaled, info_data, by.x = 'Library', by.y = 'Library')
species_rank_scaled_labeled$sample.name <- paste0(species_rank_scaled_labeled$ID, "_", species_rank_scaled_labeled$Time)
dim(species_rank_scaled_labeled)
species_rank_scaled_labeled <- species_rank_scaled_labeled[,c(2:458, 466)]
long_data_species <- species_rank_scaled_labeled %>%
  pivot_longer(!sample.name, names_to = "feature", values_to = "Norm_value")
long_data_species <- long_data_species[long_data_species$Norm_value >0 ,]




full_microbiome_Dat <- rbind.data.frame(long_data_COG, long_data_KO, long_data_PFAM, long_data_species)

#write.csv(full_microbiome_Dat, "C://Users/priceade/Desktop/Fecal_Omics/Current_Code&Workflow/Data/microbiome_full_filtered.csv", row.names = FALSE)




