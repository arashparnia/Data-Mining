##############################
#Clean Console and Environment
##############################
cat("\014") 
rm(list = ls())
##############################
# Read Training Data Set
##############################
WD<-setwd("C:\\Users\\Nikitas Marios\\Desktop\\Data Mining Techniques\\Assignment_2")
load("CleanedData_Click_Book.RData")
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
cd<-cd[c(-13)]
cd<-cbind(cd,consumer)
remove(consumer)
score2<-data.frame((1/32)*cd$click_bool+(31/32)*cd$booking_bool)
##############################
#LambdaMART Training
##############################
library("gbm")
gbm_ndcg<- gbm(score~prop_starrating+prop_review_score+prop_brand_bool+prop_location_score2+promotion_flag+norm_price_usd,
                data=cd, # dataset
                distribution=list(   # loss function:
                name='pairwise',   # pairwise
                metric="ndcg",     # ranking metric: normalized discounted cumulative gain
                group='srch_id'),    # column indicating query groups
                n.trees=2000,        # number of trees
                shrinkage=0.005,     # learning rate
                interaction.depth=5, # number per splits per tree
                bag.fraction = 1,  # subsampling fraction
                train.fraction = 1,  # fraction of data for training
                n.minobsinnode = 10, # minimum number of obs for split
                keep.data=TRUE,      # store copy of input data in model
                cv.folds=4,          # number of cross validation folds
                verbose = TRUE,     # don't print progress
                n.cores = 4)         # use a single core