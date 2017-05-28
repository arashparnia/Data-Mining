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
load("LambdaMART_P_C_v0.1.RData")
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
cd_test<-cd_test[c(-12)]
cd_test<-cbind(cd_test,consumer)
remove(consumer)
##############################
# Prediction
##############################
drops<-c("prop_location_score1","price_usd")
cd_test<-cd_test[,!(colnames(cd_test) %in% drops)]
Pclass<-cd_test$Pclass
drops<-c("Pclass")
cd_test<-cd_test[,!(colnames(cd_test) %in% drops)]
cd_test<-cbind(cd_test,data.frame(Pclass))
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
