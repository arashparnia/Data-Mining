##############################
#Clean Console and Environment
##############################
cat("\014") 
rm(list = ls())
##############################
# Read Training Data Set
##############################
WD<-setwd("C:\\Users\\Nikitas Marios\\Desktop\\Data Mining Techniques\\Assignment_2\\Data Mining VU data")
load("CleanedData_excl_comp.RData")
load("tr_comp_data.RData")
##############################
# Keep search and site id separately
##############################
id_data<-integer_data[,1:2]
integer_data<-integer_data[,3:ncol(integer_data)]
##############################
# Calculate the missing values rates for the competition variables
##############################
comp_missing_rates<-matrix(,1,ncol(competitors_data))
colnames(comp_missing_rates)<-colnames(competitors_data)
for (i in 1:ncol(competitors_data)){
comp_missing_rates[1,i]<-sum(is.na(competitors_data[,i]))/length(competitors_data[,i])
}
##############################
# Calculate the missing values rates for the factor data
##############################
factor_missing_rates<-matrix(,1,ncol(factor_data))
colnames(factor_missing_rates)<-colnames(factor_data)
for (i in 1:ncol(factor_data)){
  factor_missing_rates[1,i]<-sum(is.na(factor_data[,i]))/length(factor_data[,i])
}
##############################
# Calculate the missing values rates for the integer data
##############################
#CHECK FOR "NULL" OBSERVATIONS IN EACH OF THE VARIABLES
nullcheck<-matrix(,1,ncol(integer_data))
colnames(nullcheck)<-colnames(integer_data)
for (i in 1:ncol(integer_data)){
nullcheck[,i]<-nrow(integer_data[integer_data[,i]=="NULL",])/nrow(integer_data)
}
##############################
# Plotting
##############################
# Merge/Join the percentage results in one variable
plotdata<-cbind(factor_missing_rates,nullcheck,comp_missing_rates)
plotdata<-plotdata[,order(plotdata[1,]),drop=FALSE]
op <- par(no.readonly = TRUE)
par(mar=c(11, 4, 5, 2) + 0.3)
barplot(plotdata,ylab="Percentage of missing data",ylim=c(0,1),las=2,col=c("red"))
par(op)
