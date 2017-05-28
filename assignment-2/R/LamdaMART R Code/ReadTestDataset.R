##############################
#Clean Console and Environment
##############################
cat("\014") 
rm(list = ls())
##############################
# Read Training Data Set
##############################
WD<-setwd("C:\\Users\\Nikitas Marios\\Desktop\\Data Mining Techniques\\Assignment_2\\Data Mining VU data")
testdata<-read.csv("test_set_VU_DM_2014.csv")
