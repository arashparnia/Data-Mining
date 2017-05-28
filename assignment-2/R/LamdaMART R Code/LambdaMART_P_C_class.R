##############################
#Clean Console and Environment
##############################
cat("\014") 
rm(list = ls())
library("gbm")
##############################
# Read Training Data Set
##############################
WD<-setwd("C:\\Users\\Nikitas Marios\\Desktop\\Data Mining Techniques\\Assignment_2\\FinalTrainingAndForecasting")
#load("CleanedData.RData")
load("CleanedData2.RData")
cd<-cd2
remove(cd2)
#load("CleanedTestData.RData")
##############################
#Transform the consumer variable
##############################
cd$consumer[cd$consumer=="single"]<-1
cd$consumer[cd$consumer=="couple"]<-2
cd$consumer[cd$consumer=="Parents"]<-3
cd$consumer[cd$consumer=="other"]<-4
consumer<-data.frame(as.numeric(as.character(cd$consumer)))
names(consumer)[1]<-paste("consumer")
cd<-cd[c(-12)]
cd<-cbind(cd,consumer)
remove(consumer)
##############################
#LambdaMART training loop for each PClass (6 in total) and for each Cclass
##############################
library("gbm")
for (i in seq(1,6,1) ){
  for (j in seq(1,4,1)){
    #filter the data
    filtered_data<-cd[cd$Pclass==i&cd$consumer==j,]
    #create a name for the data that will be used in the training
    training_data_name<-paste("training_filtered_data",i,j,sep = "_")
    #pick up half of the filtered data for training
    training_filtered_data<-filtered_data[(1:round(nrow(filtered_data)/2)),]
    #assign to the name the training data
    assign(training_data_name,training_filtered_data)
    #create a name for the data that will be used in the testing
    testing_data_name<-paste("testing_filtered_data",i,j,sep="_")
    #pick up the rest half of the filtered data for testing
    testing_filtered_data<-filtered_data[((nrow(training_filtered_data)+1):nrow(filtered_data)),]
    #assign to the testing name the testing data
    assign(testing_data_name,testing_filtered_data)
  #name of the model
  LM_name <- paste("LM",i,j, sep = "_")
  #LambdaMART
  LambdaMART <- gbm(score~prop_starrating+prop_review_score+prop_location_score2,
                    +promotion_flag+norm_price_usd+random_bool,#model
                  data=training_filtered_data, # dataset
                  distribution=list(   # loss function:
                  name='pairwise',   # pairwise
                  metric="ndcg",     # ranking metric: normalized discounted cumulative gain
                  group='srch_id'),    # column indicating query groups
                  n.trees=4000,        # number of trees
                  shrinkage=0.005,     # learning rate
                  interaction.depth=5, # number per splits per tree
                  bag.fraction = 1,  # subsampling fraction
                  train.fraction = 1,  # fraction of data for training
                  n.minobsinnode = 10, # minimum number of obs for split
                  keep.data=TRUE,      # store copy of input data in model
                  cv.folds=5,          # number of cross validation folds
                  verbose = TRUE,     # don't print progress
                  n.cores = 3)         # use a single core
  assign(LM_name, LambdaMART)
}}
