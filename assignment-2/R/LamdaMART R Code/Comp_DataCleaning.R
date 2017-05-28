##############################
#Clean Console and Environment
##############################
cat("\014") 
rm(list = ls())
##############################
# Read Training Data Set
##############################
WD<-setwd("C:\\Users\\Nikitas Marios\\Desktop\\Data Mining Techniques\\Assignment_2\\Data Mining VU data")
load("competitors_data.RData")
##############################
# Transform the floating factors into numeric
##############################
for (i in 1:ncol(competitors_data)){
  competitors_data [,i] <- as.numeric(as.character(competitors_data[ ,i]))
}
##############################
# Save Transformed Competition Data
##############################
save(competitors_data, file="tr_comp_data.RData")