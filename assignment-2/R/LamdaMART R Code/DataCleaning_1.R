##############################
#Clean Console and Environment
##############################
cat("\014") 
rm(list = ls())
##############################
# Read Training Data Set
##############################
WD<-setwd("C:\\Users\\Nikitas Marios\\Desktop\\Data Mining Techniques\\Assignment_2\\Data Mining VU data")
load("rawdata_environment.RData")
##############################
#Keep separately the competitors data
##############################
competitors_data<-rawdata[,28:51]
no_comp_data<-rawdata[,c(1:27,52:54)]
##############################
#Save Competitors and Non Competitors data Separately
##############################
save(competitors_data,file="competitors_data.RData")
save(no_comp_data,file="no_comp_data.RData")