##############################
#Clean Console and Environment and Load Required Library
##############################
cat("\014") 
rm(list = ls())
library("gbm")
##############################
# Activate Working Directory
##############################
WD<-setwd("C:\\Users\\Nikitas Marios\\Desktop\\Data Mining Techniques\\Assignment_2\\FinalTrainingAndForecasting")
##############################
# Load Trained Models and Testing Dataset
##############################
load("TrainedLamdaMart.RData")
load("TestingData.RData")
cd_test<-rbind(testing_filtered_data_1_1,testing_filtered_data_1_2,testing_filtered_data_1_3,testing_filtered_data_1_4,
               testing_filtered_data_2_1,testing_filtered_data_2_2,testing_filtered_data_2_3,testing_filtered_data_2_4,
               testing_filtered_data_3_1,testing_filtered_data_3_2,testing_filtered_data_3_3,testing_filtered_data_3_4,
               testing_filtered_data_4_1,testing_filtered_data_4_2,testing_filtered_data_4_3,testing_filtered_data_4_4,
               testing_filtered_data_5_1,testing_filtered_data_5_2,testing_filtered_data_5_3,testing_filtered_data_5_4,
               testing_filtered_data_6_1,testing_filtered_data_6_2,testing_filtered_data_6_3,testing_filtered_data_6_4)
##############################
# Prediction
##############################
for (i in seq(1,6,1) ){
  for (j in seq(1,4,1)){
    Pr_name <- paste("Prediction",i,j, sep = "_") 
    
    predictionTest<-data.frame(predict(get(paste("LM",i,j,sep="_")),cd_test[cd_test$Pclass==i&cd_test$consumer==j,],
                                       gbm.perf(get(paste("LM",i,j,sep="_")),method='cv')))
    
    predictionTest<-cbind(cd_test$srch_id[cd_test$Pclass==i&cd_test$consumer==j],cd_test$prop_id[cd_test$Pclass==i&cd_test$consumer==j],predictionTest)
    names(predictionTest)[3]<-paste("Prediction",i,j,sep="_")
    names(predictionTest)[2]<-paste("prop_id")
    names(predictionTest)[1]<-paste("srch_id")
    predictionTest<-predictionTest[order(predictionTest[,1],-predictionTest[,3]),]
    assign(Pr_name, predictionTest)
  }}