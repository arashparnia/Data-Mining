##############################
#Clean Console and Environment and Load Required Library
##############################
cat("\014") 
rm(list = ls())
library("gbm")
##############################
# Activate Working Directory
##############################
WD<-setwd("C:\\Users\\Nikitas Marios\\Desktop\\Data Mining Techniques\\Assignment_2")
##############################
# Load Trained Models
##############################
load("LambdaMART_Cclass_v0.1.RData")
#load("LambdaMART_Pclass_v0.4.RData")
#load("Predictions.RData")
##############################
# Load Test Data and Change the Form of the Consumer Variable
##############################
load("CleanedTestData.RData")
cd_test$consumer[cd_test$consumer=="single"]<-1
cd_test$consumer[cd_test$consumer=="couple"]<-2
cd_test$consumer[cd_test$consumer=="Parents"]<-3
cd_test$consumer[cd_test$consumer=="other"]<-4
consumer<-data.frame(as.numeric(as.character(cd_test$consumer)))
names(consumer)[1]<-paste("consumer")
cd_test<-cd_test[c(-11)]
cd_test<-cbind(cd_test,consumer)
remove(consumer)
##############################
# Prediction
##############################  
for (i in seq(1,4,1) ){
  Pr_name <- paste("C_Prediction",i, sep = "_") 
  
  predictionTest<-data.frame(predict(get(paste("gbm_Ctype",i,sep="_")),cd_test[cd_test$consumer==i,],
                                     gbm.perf(get(paste("gbm_Ctype",i,sep="_")),method='cv')))
  
  predictionTest<-cbind(cd_test$srch_id[cd_test$consumer==i],predictionTest)
  names(predictionTest)[2]<-paste("C_Prediction",i,sep="_")
  names(predictionTest)[1]<-paste("srch_id")
  predictionTest<-predictionTest[order(predictionTest[,1],-predictionTest[,2]),]
  assign(Pr_name, predictionTest)
}