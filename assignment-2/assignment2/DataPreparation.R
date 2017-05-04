# Introduction
# In this script, code is written... :O

rm(list=ls())        # Delete workspace

######################################################################################################################
######################################################################################################################
# Installing Packages
# source("https://s3-us-west-2.amazonaws.com/10x.files/supp/cell-exp/rkit-install-1.1.0.R")
# install.packages("cellrangerRkit")

# Library
library("ggplot2")
# library(cellrangerRkit)
# packageVersion("cellrangerRkit")
######################################################################################################################
######################################################################################################################
## DATA LOADING

# Read Data (sorted on time)
Data_Raw <- read.csv("C:/Users/Piet/Documents/Assignment2/training_set_VU_DM_2014.csv", header= TRUE,nrows=20000)


WD <- setwd("C:\\Users\\Piet\\Documents\\Assignment2")
load("DD.RData")


# Remove duplicates [data is already unique]
Data_Raw <- unique(Data_Raw)



# Sample 25% data from Data_Raw
# NOTE PASSENGER ID NOT COMPLETE NOW
Data_Raw.sample <- Data_Raw[sample(1:nrow(Data_Raw), nrow(Data_Raw)*0.25),]


# Group by:

# ddply(Data_Raw,~Data_Raw$srch_id,summarise)

######################################################################################################################
######################################################################################################################
## DATA ANALYSIS

# Every srch_id represent 1 customer  [work with cell/list data]

Data_Raw.list <-  split(Data_Raw, Data_Raw$srch_id)




######################################################################################################################
######################################################################################################################



## DATA PLOTTING

summary(Data_Raw)

table(Data_Raw[ ,6])



######################################################################################################################
######################################################################################################################
## SUBSETTING THE DATA

# 1. Customer Information
Data_customer <- Data_Raw[ , 1:6]
Data_customer <- unique(Data_customer)
customers <- data.frame(Data_customer[, 1])
customers <- unique(customers)

# 2. Hotel Information
Data_hotel <- Data_Raw[ , c(1, 7:17)]
Data_hotel <- unique(Data_hotel)

# 3. Search Information
Data_search <- Data_Raw[ , c(1, 18:24) ]
Data_search <- unique(Data_search)

# 4. Probability Information
Data_prob <- Data_Raw[ , c(1, 25:27) ]
Data_prob <- unique(Data_prob)

# 5. Competition Information
Data_comp <- Data_Raw[ , c(1, 28:51) ]
Data_comp <- unique(Data_comp)

# 5. Information ONLY on training set
Data_training <- Data_Raw[ , c(1, 52:54) ]
Data_training <- unique(Data_training)

######################################################################################################################
######################################################################################################################
# Decision tree subgroups Customers



# Split Previous and Non-Previous customer's
customers_NonRaters<- data.frame(Data_customer$srch_id[Data_customer$visitor_hist_starrating == 'NULL'])
customers_Raters<- data.frame(Data_customer$srch_id[Data_customer$visitor_hist_starrating != 'NULL'])

customers_NonBookers <- data.frame(Data_customer$srch_id[Data_customer$visitor_hist_adr_usd == 'NULL'])
customers_Bookers<- data.frame(Data_customer$srch_id[Data_customer$visitor_hist_adr_usd != 'NULL'])

######################################################################################################################
######################################################################################################################
# Missing values


######################################################################################################################
######################################################################################################################
# Handling outliers



######################################################################################################################
######################################################################################################################
# Scatter plots

# Customer location & Hotel location
plot(Data_Raw$visitor_location_country_id, Data_Raw$prop_country_id, main="Scatterplot Example", 
     xlab="Location consumer", ylab="Location hotel", pch=19)





