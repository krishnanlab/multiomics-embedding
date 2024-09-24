#set path to folder with node embedding and sample data
path <- "C:/Users/priceade/Desktop/multiomics-embedding/base_end_classifiers/"

# Load required packages
library(dplyr)
library(stringr)
library(tidyverse)
library(glmnet)
library(caret)
library(gt)
library(here)


# Reading matrix output from pecan py, using all data and 128 dimension parameter
pecan_matrix <- read.table(file = paste0(path, "data_128_dim.txt"), fill = TRUE)
col_name<- paste("dim_", 1:128, sep = "")
colnames_matrix <- c("nodes",col_name)
#remove first row of node network b/z this is just the dim of the matrix
pecan_matrix_128 <- pecan_matrix[-1,]
colnames(pecan_matrix_128) <- colnames_matrix



#add label to each row
pecan_matrix_128$category <- ifelse(str_detect(pecan_matrix_128$nodes, "MD-"),"Baby",
                                    ifelse(str_detect(pecan_matrix_128$nodes,"SPE|PF|K|COG|ENOG"),"Microbiome","Metabolomic"))
pecan_matrix_128$point <- ifelse(str_detect(pecan_matrix_128$nodes,"k_"),"Microbe_Species","check")
pecan_matrix_128$point <- ifelse(str_detect(pecan_matrix_128$nodes, "PF"),"Microbe_PFAM",pecan_matrix_128$point)
pecan_matrix_128$point <- ifelse(startsWith(pecan_matrix_128$nodes, "K"),"Microbe_K0",pecan_matrix_128$point)
pecan_matrix_128$point <- ifelse(str_detect(pecan_matrix_128$nodes, "COG|ENOG"),"Microbe_COG",pecan_matrix_128$point)
pecan_matrix_128$point <- ifelse(str_detect(pecan_matrix_128$nodes, "Base"),"Baby_base",pecan_matrix_128$point)
pecan_matrix_128$point <- ifelse(str_detect(pecan_matrix_128$nodes, "End"),"Baby_end",pecan_matrix_128$point)
pecan_matrix_128$point <- ifelse(str_detect(pecan_matrix_128$nodes, "N_LP"),"meta_neg_lipid",pecan_matrix_128$point)
pecan_matrix_128$point <- ifelse(str_detect(pecan_matrix_128$nodes, "P_LP"),"meta_pos_lipid",pecan_matrix_128$point)
pecan_matrix_128$point <- ifelse(str_detect(pecan_matrix_128$nodes, "P_AQ"),"meta_pos_aq",pecan_matrix_128$point)
pecan_matrix_128$point <- ifelse(str_detect(pecan_matrix_128$nodes, "N_AQ"),"meta_neg_aq",pecan_matrix_128$point)




#Read in info for samples
info_data <- read_csv(paste0(path, "microbiome_info_data.csv"))
colnames(info_data)[11]<-"nodes"
#subset and reorder info data to begin with node name (just removes column of row numbers)
info_data<- info_data[,c(11,2:10)]
#remove unknown timepoints annotates with ?? *** 2 values removed ***
info_data<-info_data[info_data$Timepoint != "??",]


#######################prepare data for training and testing split#############
baby_Pecan_128 <- inner_join(info_data,pecan_matrix_128, by = join_by(nodes))
dim(baby_Pecan_128)#dimension is 109, contains merged microbiome info data and predicted node network data
#filter and rearrange data to have nodes, ID, group, timepoint, and all pecan network predictions
baby_Pecan_128_u<-  baby_Pecan_128[,c(1,3,4,5,11:138)]
#make column combining group (dairy or meat) and timepoint var (baseline or late obs)
baby_Pecan_128_u$new_var <-paste(baby_Pecan_128_u$Group,baby_Pecan_128_u$Timepoint,sep="_")
#keep new combined group for all late obs, but set all baseline obs (both dairy and meat) as just BL
baby_Pecan_128_u$new_var <- ifelse(str_detect(baby_Pecan_128_u$new_var,"BL"),"BL",baby_Pecan_128_u$new_var)
#make a new column holding previously combined group and timepoint as factors (3 different factors)
baby_Pecan_128_u$diet_time <- as.factor(baby_Pecan_128_u$new_var)
#make new dataframe containing nodes, baby ID, combined group and timepoint as factors (3 different factors), and predicted network
baby_Pecan_128_v <- baby_Pecan_128_u[,c(1,2,134,5:132)] 



#################isolate meat and dairy groups for balanced random sampling###############
#baby pecan dairy now only contains dairy diet end timepoints
baby_pecan_dairy <- baby_Pecan_128_v[baby_Pecan_128_v$diet_time == "Dairy_E",]
#baby pecan meat now only contains meat diet end timepoints
baby_pecan_meat <- baby_Pecan_128_v[baby_Pecan_128_v$diet_time == "Meat_E",]
#make list of baby IDs for diary end time points
dairy_list <- (baby_pecan_dairy$ID) #29 (27 also in baseline)
#make list of baby IDs for meat end time points
meat_list <- (baby_pecan_meat$ID) #23 (21 also in baseline)




##########5 models: Randomly sample data into 5, with 5 training on 80% and 5 test on 20%

#set up empty data frames to hold model performance results
accuracy_cv_baseline <- data.frame(matrix(ncol=3, nrow=5))
by_class_Acc_cv_baseline<- data.frame(matrix(ncol=2, nrow=5))
by_class_F1_cv_baseline <- data.frame(matrix(ncol=2, nrow=5))
colnames(accuracy_cv_baseline) <- c("fold","overall_Baseline","Baseline_Lambda")
colnames(by_class_Acc_cv_baseline) <- c("fold","balanced_Base_not_base")
colnames(by_class_F1_cv_baseline) <- c("fold","F1_Base_not_base")
meta_conf_cv_baseline <-data.frame(matrix(ncol=7, nrow=4)) 



###############balanced random sampling of training and testing for 5 models######
set.seed(125)
dairy_1 <- sample(dairy_list,6)
meat_1 <- sample(meat_list,5)
test_1 <- c(dairy_1,meat_1)

dairy_2 <- sample(dairy_list[!dairy_list %in% dairy_1],6)
meat_2 <- sample(meat_list[!meat_list %in% meat_1],5)
test_2 <-c(dairy_2, meat_2)

dairy_3 <- sample(dairy_list[!dairy_list %in% c(dairy_1,dairy_2)],6)
meat_3 <- sample(meat_list[!meat_list %in% c(meat_1, meat_2)],5)
test_3 <-c(dairy_3, meat_3)

dairy_4 <- sample(dairy_list[!dairy_list %in% c(dairy_1,dairy_2,dairy_3)],5)
meat_4 <- sample(meat_list[!meat_list %in% c(meat_1, meat_2, meat_3)],4)
test_4 <-c(dairy_4, meat_4)

dairy_5 <- dairy_list[!dairy_list %in% c(dairy_1,dairy_2,dairy_3, dairy_4)]
meat_5 <- meat_list[!meat_list %in% c(meat_1, meat_2, meat_3, meat_4)]
test_5 <-c(dairy_5, meat_5)

Partitioned_data <- data.frame(matrix(ncol=2, nrow=52))
colnames(Partitioned_data) <- c("BabyID","Test_n")
Partitioned_data[1:11,1] <- test_1
Partitioned_data[1:11,2] <- "test_1"
Partitioned_data[12:22,1] <- test_2
Partitioned_data[12:22,2] <- "test_2"
Partitioned_data[23:33,1] <- test_3
Partitioned_data[23:33,2] <- "test_3"
Partitioned_data[34:42,1] <- test_4
Partitioned_data[34:42,2] <- "test_4"
Partitioned_data[43:52,1] <- test_5
Partitioned_data[43:52,2] <- "test_5"




###############function for training and testing a single baseline/endpoint classifier###########
compute_ML <- function(data, test_data) {
  #testing data
  ##########predict baseline or endpoint from network testing data######
  test_data_baby <- data[data$ID %in%test_data,]
  test_data_baseline <- test_data_baby
  test_data_baseline$diet_time <- ifelse(test_data_baseline$diet_time == "BL","BL","NOT_BL")
  test_data_baseline_x <- test_data_baseline[,c(4:131)]
  test_data_baseline_y <- test_data_baseline[,c(3)]
  
  
  #training data
  ###baseline, predicting BL or NOT_BL
  train_1_baby <- data[!data$ID %in%test_data,]
  train_1_baby_baseline <- train_1_baby
  train_1_baby_baseline$diet_time <- ifelse(train_1_baby_baseline$diet_time == "BL","BL","NOT_BL")
  train_1_baby_baseline_x <- train_1_baby_baseline[,c(4:131)]
  train_1_baby_baseline_y <- train_1_baby_baseline[,c(3)]
  #2 levels of outcome, BL & NOT_BL
  y_1_baseline <- as.factor(train_1_baby_baseline_y$diet_time)
  matrix_x_1_baseline <- as.matrix(train_1_baby_baseline_x)

  
  #############################Model training###############################################
  #baseline, predicting BL or NOt_BL
  cvfit_1_baseline <- cv.glmnet(matrix_x_1_baseline, y_1_baseline, family = "binomial", type.measure = 'class')
  lambda_used_1_baseline<- cvfit_1_baseline$lambda.min
  
 
  #####################################Model Testing##########################################
  #Predicting baseline, predicting BL or NOT_BL
  matrix_x_test_data_baseline <- as.matrix(test_data_baseline_x)
  actual_y_1_baseline <- as.factor(test_data_baseline$diet_time)
  prediction<- (predict(cvfit_1_baseline, newx = matrix_x_test_data_baseline, s = "lambda.min", type = "class"))
  prediction_a_1_baseline <- as.factor(prediction)
  levels(prediction_a_1_baseline) <-  levels(y_1_baseline)
  
  
  #Creating confusion matrix for baseline/endpoint classifier
  conf_matrix_1_baseline <- confusionMatrix(data=prediction_a_1_baseline, reference = actual_y_1_baseline,dnn = c("Prediction", "Reference"))
  
  return(list(conf_matrix_1_baseline,lambda_used_1_baseline))
}








# make a list of test data for all 5 models
test_list <- list(test_1,test_2,test_3,test_4,test_5)

#loop to build models and get performance stats for each
for (i in 1:5) {
  test <- compute_ML(baby_Pecan_128_v, test_list[[i]])
  
  all_category <- test[[1]]
  lambda <-test[[2]] 
  
  #Accuracy
  accuracy_cv_baseline[i,1] <- i
  accuracy_cv_baseline[i,2] <- all_category$overall[['Accuracy']]
  accuracy_cv_baseline[i,3] <- lambda
  
  #Balanced Accuracy 
  by_class_Acc_cv_baseline[i,1] <- i
  by_class_Acc_cv_baseline[i,2] <- all_category$byClass[11]
  
  #F1 Score 
  by_class_F1_cv_baseline[i,1] <- i
  by_class_F1_cv_baseline[i,2] <- all_category$byClass[7]

}




########format base/end classifier performance results for all 5 models######
final_overall_accuracy <-  accuracy_cv_baseline[,1:3]
colnames(final_overall_accuracy) <- c("model","Base_v_endpoint", "lamda_used")
final_overall_accuracy <- final_overall_accuracy %>% 
  ungroup %>% 
  add_row(!!! colMeans(.[-1]))
final_overall_accuracy[6,1] <- "MEAN"

final_balanced_accuracy <- by_class_Acc_cv_baseline[,1:2]
colnames(final_balanced_accuracy) <- c("model","Base_v_endpoint")
final_balanced_accuracy <- final_balanced_accuracy %>% 
  ungroup %>% 
  add_row(!!! colMeans(.[-1]))
final_balanced_accuracy[6,1] <- "MEAN"

final_f1_score <- by_class_F1_cv_baseline
colnames(final_f1_score) <- c("model","Base_v_endpoint")
final_f1_score <- final_f1_score %>% 
  ungroup %>% 
  add_row(!!! colMeans(.[-1]))
final_f1_score[6,1] <- "MEAN"

full_results <- cbind.data.frame('Model' = final_overall_accuracy[,1], 'lambda used' = round(final_overall_accuracy[,3], 4),
                                 'Accuracy' =  final_overall_accuracy[,2],
                                 'Balanced Accuracy' = final_balanced_accuracy[,2], 'F1 Score'=final_f1_score[,2])

full_results[6,2] <- '--'


gt(full_results)







