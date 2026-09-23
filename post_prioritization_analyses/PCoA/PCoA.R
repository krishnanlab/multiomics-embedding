library(vegan)
library(ecodist)
library(metagMisc)
library(stringr)
library(ggplot2)






#################PCoA for ranking normalized data########################
full_data <- read.csv("C:/Users/priceade/Desktop/Fecal_Omics/Current_Code&Workflow/Data/microbe_metabolites_filtered_rank_normalized.csv")
samples <- read.csv("C:/Users/priceade/Desktop/Fecal_Omics/Older Code and Workflows/samples_for_analysis.csv")
samples_parsed <- cbind.data.frame("ID" = samples$ID, "timepoint_group" = ifelse(samples$Time == "End", paste0("E_", samples$Group), "BL"), "sample.name" = samples$sample.name)
full_data_grouptimepoint <- cbind.data.frame(full_data, samples_parsed)
full_data_grouptimepoint[is.na(full_data_grouptimepoint)] <- 0
rm(full_data)



#################PCoA for network embedding########################
library(text2vec)
# Reading matrix output from pecan py, using all data and 128 dimension parameter
pecan_matrix <- read.table(file = "C:/Users/priceade/Desktop/Fecal_Omics/Current_Code&Workflow/top_embeddings/ai9n4jxs.tsv.gz", fill = TRUE)
#other embedding spaces
#pecan_matrix <- read.table(file = "C:/Users/priceade/Desktop/Fecal_Omics/Current_Code&Workflow/top_embeddings/7o4yga2v.tsv.gz", fill = TRUE)

colnames(pecan_matrix)<- (paste("dim_", 1:128, sep = ""))

#remove first row of node network b/z this is just the dim of the matrix
pecan_matrix_128 <- pecan_matrix
pecan_matrix_128$nodes <- rownames(pecan_matrix_128)
pecan_matrix_128 <- pecan_matrix_128[,c(129, 1:128)]

pecan_matrix_128$category <- ifelse(str_detect(pecan_matrix_128$nodes, "MD-"),"Baby",
                                    ifelse(str_detect(pecan_matrix_128$nodes,"SPE|PF|K|COG|ENOG"),"Microbiome","Metabolomic"))

get_pc_baby_unordered <- pecan_matrix_128[pecan_matrix_128$category == "Baby",]
  

get_pc_baby_ordered<- get_pc_baby_unordered[order(get_pc_baby_unordered$nodes),]
get_pc_baby <- get_pc_baby_ordered[,2:129]
sample.groups_time_type = full_data_grouptimepoint$timepoint_group

####################################Compare across BL, Dairy_E, Meat_E###############
dist <- round(1- sim2(as.matrix(get_pc_baby), method = "cosine", norm = "l2"), 15)
pco <- wcmdscale(dist, eig = TRUE)
full_dat <- cbind.data.frame(pco$points[,1:2], sample.groups_time_type)


############PERMANOVA#############
dm <-dist
full_mat <- cbind.data.frame(sample.groups_time_type, dm)

# PERMANOVA TEST ---
set.seed(12345)
y_permanova <- vegan::adonis2(dm ~ sample.groups_time_type,  data=full_mat, permutations=10000, method="cosine")
p <- y_permanova$`Pr(>F)`[1]


ggplot(full_dat, aes(x=Dim1, y=Dim2, color=sample.groups_time_type)) +
  geom_point(size = 2.5, shape= 'circle') + theme_gray() +
  theme(axis.line = element_line(colour = "black"), 
        panel.border = element_rect(color = "black", fill = NA, size = 1), 
        axis.text=element_text(size=12),
        axis.title=element_text(size=14,face="bold")) +
  xlab(paste0("Dim 1 (", round(100*(pco$eig[1]/sum(pco$eig)), 1), "%)")) +
  ylab(paste0("Dim 2 (", round(100*(pco$eig[2]/sum(pco$eig)), 1), "%)")) +
  scale_colour_manual(values = c("darkorange","forestgreen","maroon")) +
  ggtitle(paste0("PCoA of sample network embeddings, R^2=", round(y_permanova$R2[1], 3), '/n p=', round(p,6)))+
  stat_ellipse(size=1) 

#######################Compare across BL, Endpoint#######################
sample.groups_time_type <- ifelse(str_detect(sample.groups_time_type, "BL"),"BL", "End")
full_dat <- cbind.data.frame(pco$points[,1:2], sample.groups_time_type)
############PERMANOVA#############
dm <- dist

full_mat <- cbind.data.frame(sample.groups_time_type, dm)

# PERMANOVA TEST ---
set.seed(12345)
y_permanova <- vegan::adonis2(dm ~ sample.groups_time_type,  data=full_mat, permutations=10000, method="cosine")
p <- y_permanova$`Pr(>F)`[1]


ggplot(full_dat, aes(x=Dim1, y=Dim2, color=sample.groups_time_type)) +
  geom_point(size = 2.5, shape= 'circle') + theme_gray() +
  theme(axis.line = element_line(colour = "black"), 
        panel.border = element_rect(color = "black", fill = NA, size = 1), 
        axis.text=element_text(size=12),
        axis.title=element_text(size=14,face="bold")) +
  xlab(paste0("Dim 1 (", round(100*(pco$eig[1]/sum(pco$eig)), 1), "%)")) +
  ylab(paste0("Dim 2 (", round(100*(pco$eig[2]/sum(pco$eig)), 1), "%)")) +
  scale_colour_manual(values = c("darkorange","darkblue")) +
  ggtitle(paste0("PCoA of sample network embeddings, R^2=", round(y_permanova$R2[1], 3), '/n p=', round(p,6)))+
  stat_ellipse(size=1) + xlim(-.43, .43) 


print(paste('r2', y_permanova$R2[1]))
print(paste('pval', p))

#######################Compare across Dairy_E, Meat_E################################
get_pc_baby_ordered$type <- sample.groups_time_type

get_pc_baby_ordered_ends <- get_pc_baby_ordered[get_pc_baby_ordered$type != "BL",]

get_pc_type <- get_pc_baby_ordered_ends[,2:129]

dist <- round(1- sim2(as.matrix(get_pc_type), method = "cosine", norm = "l2"), 15)
pco <- wcmdscale(dist, eig = TRUE)
## plot
all_groups = full_data_grouptimepoint$timepoint_group
sample.groups_type = all_groups[all_groups != 'BL']
full_dat <- cbind.data.frame(pco$points[,1:2], sample.groups_type)


###########PERMANOVA for last ex##################
# GET DISTANCE MATRIX ---
dm <-dist

full_mat <- cbind.data.frame(sample.groups_type, dm)

# PERMANOVA TEST ---
set.seed(12345)
y_permanova <- vegan::adonis2(dm ~ sample.groups_type,  data=full_mat, permutations=10000, method="cosine")
p <- y_permanova$`Pr(>F)`[1]


ggplot(full_dat, aes(x=Dim1, y=Dim2, color=sample.groups_type)) +
  geom_point(size = 2.5, shape= 'circle') + theme_gray() +
  theme(axis.line = element_line(colour = "black"), 
        panel.border = element_rect(color = "black", fill = NA, size = 1), 
        axis.text=element_text(size=12),
        axis.title=element_text(size=14,face="bold")) +
  xlab(paste0("Dim 1 (", round(100*(pco$eig[1]/sum(pco$eig)), 1), "%)")) +
  ylab(paste0("Dim 2 (", round(100*(pco$eig[2]/sum(pco$eig)), 1), "%)")) +
  scale_colour_manual(values = c("forestgreen","maroon")) +
  ggtitle(paste0("PCoA of sample network embeddings, R^2=", round(y_permanova$R2[1], 3), '/n p=', round(p,6)))+
  stat_ellipse(size=1) + xlim(-.43, .43) 


print(paste('r2', y_permanova$R2[1]))
print(paste('pval', p))










###################################combined microbiome and metabolome################################



#################PCoA for ranking normalized data########################
#full_data_grouptimepoint <- read.csv("C:/Users/priceade/Desktop/Fecal_Omics/processed_data_1/Regression Model Data/groupxtime_labeledsamples_microbiome_metabolome_data.csv")

library(ggplot2)
##############################METABOLOME####################################
get_pc_baby <- full_data_grouptimepoint[,2:25901]
####################################Compare across BL, Dairy_E, Meat_E###############
sample.groups_time_type = full_data_grouptimepoint$timepoint_group


############PERMANOVA#############
dist <- round(1- sim2(as.matrix(get_pc_baby), method = "cosine", norm = "l2"), 15)
pco <- wcmdscale(dist, eig = TRUE)
full_dat <- cbind.data.frame(pco$points[,1:2], sample.groups_time_type)


############PERMANOVA#############
dm <-dist
full_mat <- cbind.data.frame(sample.groups_time_type, dm)

# PERMANOVA TEST ---
set.seed(12345)
y_permanova <- vegan::adonis2(dm ~ sample.groups_time_type,  data=full_mat, permutations=10000, method="cosine")
p <- y_permanova$`Pr(>F)`[1]

ggplot(full_dat, aes(x=Dim1, y=Dim2, color=sample.groups_time_type)) +
  geom_point(size = 2.5, shape= 'circle') + theme_gray() +
  theme(axis.line = element_line(colour = "black"), 
        panel.border = element_rect(color = "black", fill = NA, linewidth = 1), 
        axis.text=element_text(size=12),
        axis.title=element_text(size=14,face="bold")) +
  xlab(paste0("Dim 1 (", round(100*(pco$eig[1]/sum(pco$eig)), 1), "%)")) +
  ylab(paste0("Dim 2 (", round(100*(pco$eig[2]/sum(pco$eig)), 1), "%)")) +
  scale_colour_manual(values = c("darkorange","forestgreen","maroon")) +
  ggtitle(paste0("PCoA of rank normalized features, R^2=", round(y_permanova$R2[1], 3), '/n p=', round(p,6)))+
  stat_ellipse(size=1) + xlim(-.32, .32) 


#######################Compare across BL, Endpoint#######################
sample.groups_time_type <- ifelse(str_detect(sample.groups_time_type, "BL"),"BL", "End")
full_dat <- cbind.data.frame(pco$points[,1:2], sample.groups_time_type)
full_dat$sample <- full_data_grouptimepoint$ID
############PERMANOVA#############
dist <- round(1- sim2(as.matrix(get_pc_baby), method = "cosine", norm = "l2"), 15)
pco <- wcmdscale(dist, eig = TRUE)
full_dat <- cbind.data.frame(pco$points[,1:2], sample.groups_time_type)


############PERMANOVA#############
dm <-dist
full_mat <- cbind.data.frame(sample.groups_time_type, dm)

# PERMANOVA TEST ---
set.seed(12345)
y_permanova <- vegan::adonis2(dm ~ sample.groups_time_type,  data=full_mat, permutations=10000, method="cosine")
p <- y_permanova$`Pr(>F)`[1]



ggplot(full_dat, aes(x=Dim1, y=Dim2, color=sample.groups_time_type)) +
  geom_point(size = 2.5, shape= 'circle') + theme_gray() +
  theme(axis.line = element_line(colour = "black"), 
        panel.border = element_rect(color = "black", fill = NA, linewidth = 1), 
        axis.text=element_text(size=12),
        axis.title=element_text(size=14,face="bold")) +
  xlab(paste0("Dim 1 (", round(100*(pco$eig[1]/sum(pco$eig)), 1), "%)")) +
  ylab(paste0("Dim 2 (", round(100*(pco$eig[2]/sum(pco$eig)), 1), "%)")) +
  scale_colour_manual(values = c("darkorange","darkblue")) +
  ggtitle(paste0("PCoA of rank normalized features, R^2=", round(y_permanova$R2[1], 3), '/n p=', round(p,6)))+
  stat_ellipse(size=1) + xlim(-.3, .3) 


#######################Compare across Dairy_E, Meat_E################################
pca_baby_type = full_data_grouptimepoint[which(full_data_grouptimepoint$timepoint_group == "E_Dairy" | full_data_grouptimepoint$timepoint_group == "E_Meat"),]
all_groups = full_data_grouptimepoint$timepoint_group
sample.groups_type = all_groups[all_groups != 'BL']

get_pc_type <- pca_baby_type[,2:25901]



###########PERMANOVA for last ex##################
dist <- round(1- sim2(as.matrix(get_pc_type), method = "cosine", norm = "l2"), 15)
pco <- wcmdscale(dist, eig = TRUE)
full_dat <- cbind.data.frame(pco$points[,1:2], sample.groups_type)


############PERMANOVA#############
dm <-dist
full_mat <- cbind.data.frame(sample.groups_type, dm)

# PERMANOVA TEST ---
set.seed(12345)
y_permanova <- vegan::adonis2(dm ~ sample.groups_type,  data=full_mat, permutations=10000, method="cosine")
p <- y_permanova$`Pr(>F)`[1]

ggplot(full_dat, aes(x=Dim1, y=Dim2, color=sample.groups_type)) +
  geom_point(size = 2.5, shape= 'circle') + theme_gray() +
  theme(axis.line = element_line(colour = "black"), 
        panel.border = element_rect(color = "black", fill = NA, linewidth = 1), 
        axis.text=element_text(size=12),
        axis.title=element_text(size=14,face="bold")) +
  xlab(paste0("Dim 1 (", round(100*(pco$eig[1]/sum(pco$eig)), 1), "%)")) +
  ylab(paste0("Dim 2 (", round(100*(pco$eig[2]/sum(pco$eig)), 1), "%)")) +
  scale_colour_manual(values = c("forestgreen","maroon")) +
  ggtitle(paste0("PCoA of rank normalized features, R^2=", round(y_permanova$R2[1], 3), ', p=', round(p,2)))+
  stat_ellipse(size=1) + xlim(-.3, .3) 



