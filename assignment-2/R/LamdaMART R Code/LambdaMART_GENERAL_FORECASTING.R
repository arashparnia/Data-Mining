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
load("LambdaMART_GENERAL.RData")
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
drops<-c("prop_location_score1","price_usd")
cd_test<-cd_test[,!(colnames(cd_test) %in% drops)]
Pclass<-cd_test$Pclass
drops<-c("Pclass")
cd_test<-cd_test[,!(colnames(cd_test) %in% drops)]
cd_test<-cbind(cd_test,data.frame(Pclass))

    predictions<-data.frame(predict(gbm_ndcg,cd_test,gbm.perf(gbm_ndcg,method='cv')))
    
    predictions<-cbind(cd_test$srch_id,predictions)
    names(predictions)[2]<-paste("Predictions")
    names(predictions)[1]<-paste("srch_id")
    predictions<-predictions[order(predictions[,1],-predictions[,2]),]

