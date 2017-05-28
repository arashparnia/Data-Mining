##############################
#Clean Console and Environment
##############################
cat("\014") 
rm(list = ls())
##############################
# Read Training Data Set
##############################
WD<-setwd("C:\\Users\\Nikitas Marios\\Desktop\\Data Mining Techniques\\Assignment_2")
load("CleanedData.RData")
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
cd<-cd[c(-11)]
cd<-cbind(cd,consumer)
remove(consumer)
##############################
#LambdaMART training loop for consumer type (4 in total)
##############################
library("gbm")
for (i in seq(1,4,1) ){
Cname <- paste("gbm_Ctype", i, sep = "_") #Ctype stands for consumer type
gbm_ndcg_CONS<- gbm(score~prop_starrating+prop_review_score+prop_brand_bool+prop_location_score1+prop_location_score2+prop_log_historical_price+price_usd+promotion_flag+random_bool+norm_price_usd, # formula
                          data=cd[cd$consumer==i,], # dataset
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
                          verbose = FALSE,     # don't print progress
                          n.cores = 3)         # use a single core
assign(Cname, gbm_ndcg_CONS)
}