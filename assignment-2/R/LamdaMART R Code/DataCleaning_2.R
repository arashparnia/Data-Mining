##############################
#Clean Console and Environment
##############################
cat("\014") 
rm(list = ls())
##############################
# Read Training Data Set
##############################
WD<-setwd("C:\\Users\\Nikitas Marios\\Desktop\\Data Mining Techniques\\Assignment_2\\Data Mining VU data")
load("no_comp_data.RData")
##############################
# Separate the data further into integer and floating and id variables
##############################
factor_col<-c(2,5,6,10,12:14,16,25,26,29)
factor_data<-subset(no_comp_data,select=factor_col)
integer_data<-subset(no_comp_data,select=-factor_col)
datetime<-factor_data$date_time #keep date/time variable separately
factor_data<-factor_data[,2:ncol(factor_data)] #keep all variables except date/time
##############################
# Transform the floating factors into numeric
##############################
for (i in 1:ncol(factor_data)){
  factor_data[,i] <- as.numeric(as.character(factor_data[ ,i]))
}
##############################
# SAVE THE FORMED DATA 
##############################
save(factor_data,integer_data,datetime,file="CleanedData_excl_comp.RData")
  