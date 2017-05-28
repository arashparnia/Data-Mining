##############################
#Clean Console and Environment
##############################
cat("\014") 
rm(list = ls())
##############################
# Read Training Data Set
##############################
WD<-setwd("C:\\Users\\Nikitas Marios\\Desktop\\Data Mining Techniques\\Assignment_2")
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
# Plotting Missing Percentages
##############################
# Merge/Join the percentage results in one variable
plotdata<-cbind(factor_missing_rates,nullcheck,comp_missing_rates)
plotdata<-plotdata[,order(plotdata[1,]),drop=FALSE]
op <- par(no.readonly = TRUE)
par(mar=c(11, 4, 5, 2) + 0.3)
barplot(plotdata,ylab="Percentage of missing data",ylim=c(0,1),las=2,col=c("red"))
par(op)
##############################
# Check Missing Rates for the NON Competitors variables
##############################
plotdata2<-cbind(factor_missing_rates,nullcheck)
plotdata2<-plotdata2[,order(plotdata2[1,]),drop=FALSE]
op <- par(no.readonly = TRUE)
par(mar=c(11, 4, 5, 2) + 0.3)
barplot(plotdata2,ylab="Percentage of missing data",ylim=c(0,1),las=2,col=c("red"))
par(op)
##############################
  # HISTOGRAMS
##############################
par(mfrow = c(2, 3))
hist(factor_data$price_usd)
hist(factor_data$prop_location_score2)
hist(integer_data$srch_length_of_stay)
hist(integer_data$srch_booking_window)
hist(factor_data$orig_destination_distance)
##############################
# Replacing missing observations of the variable prop_location_score2
##############################
factor_data$prop_location_score2[is.na(factor_data$prop_location_score2)]<-mean(factor_data$prop_location_score2[(!is.na(factor_data$prop_location_score2))])
##############################
# Replace missing values of prop review score with the lowest rating value (1)
##############################
factor_data$prop_review_score[is.na(factor_data$prop_review_score)] <- 1
##############################
# Outliers treat for the price_usd variables
##############################
factor_data$price_usd[factor_data$price_usd>2000] <- median(factor_data$price_usd)
##############################
# ANOVA regression classification for the prop_starrating
##############################
ANOVA_Sample<-cbind(factor_data$prop_review_score,factor_data$price_usd,integer_data$prop_starrating)
library("rpart")
ANOVA_Sample<-data.frame(ANOVA_Sample)
colnames(ANOVA_Sample)<-c("prop_review_score","price_usd","prop_starrating")
starFIT <- rpart(prop_starrating ~ prop_review_score+price_usd,data = ANOVA_Sample[ANOVA_Sample$prop_starrating!=0,], method = "anova")
ANOVA_Sample$prop_starrating[ANOVA_Sample$prop_starrating==0] <- round(predict(starFIT,data=data.frame(ANOVA_Sample[ANOVA_Sample$prop_starrating==0,],)))
#replace ald starrating varaible with the new predicted one
integer_data$prop_starrating<-ANOVA_Sample$prop_starrating

##############################
##############################
# CREATE PLOTS FOR THE EXPLORATORY DATA ANALYSIS
##############################
##############################
buckets<-c(0,100,200,300,400,500,600,700,800,2000)
Bins<-.bincode(factor_data$price_usd,buckets,TRUE,TRUE)
Bins<-data.frame(Bins)
#percentage of clicks
cr1<-length(integer_data$click_bool[integer_data$click_bool==1])/length(integer_data$click_bool)
#percentage of non clicks
cr2<-length(integer_data$click_bool[integer_data$click_bool==0])/length(integer_data$click_bool)
#percentage of books
br1<-length(integer_data$booking_bool[integer_data$booking_bool==1])/length(integer_data$booking_bool)
#percentage of non books
br2<-length(integer_data$booking_bool[integer_data$booking_bool==0])/length(integer_data$booking_bool)
#rate of books per sum of clicks
bc1<-length(integer_data$booking_bool[integer_data$booking_bool==1])/length(integer_data$click_bool[integer_data$click_bool==1])
#rate of non books per sum of clicks
bc2<-length(integer_data$booking_bool[integer_data$booking_bool==0&integer_data$click_bool==1])/length(integer_data$click_bool[integer_data$click_bool==1])
#PRICE TABLE FORMATION
price_table<-cbind(Bins,integer_data$click_bool,integer_data$booking_bool)
names(price_table)[2]<-paste("click_bool")
names(price_table)[3]<-paste("booking_bool")

#click book rates per pricebucket
pr_cbr<-matrix(,9,2)

for (i in (1:length(sort(unique(Bins[,]))))){
  pr_cbr[i,1]<-length(price_table$booking_bool[price_table$booking_bool==1& price_table$click_bool==1 & price_table$Bins==i])/length(price_table$click_bool[price_table$click_bool==1&price_table$Bins==i])
  pr_cbr[i,2]<-length(price_table$booking_bool[price_table$click_bool==1 & price_table$Bins==i])/length(price_table$click_bool[price_table$Bins==i])
}
pr_cbr<-data.frame(pr_cbr)


#click rates per star rating
star_table<-cbind(integer_data$prop_starrating,factor_data$prop_review_score,integer_data$click_bool,integer_data$booking_bool)
star_table<-data.frame(star_table)
names(star_table)[1]<-paste("prop_starrating")
names(star_table)[2]<-paste("prop_review_score")
names(star_table)[3]<-paste("click_bool")
names(star_table)[4]<-paste("booking_bool")
st_br<-matrix(,5,2)
for(i in (1:length(unique(integer_data$prop_starrating)))){
  st_br[i,1]<-length(star_table$booking_bool[star_table$booking_bool==1& star_table$click_bool==1 & star_table$prop_starrating==i])/length(star_table$click_bool[star_table$click_bool==1& star_table$prop_starrating==i])
  st_br[i,2]<-length(star_table$click_bool[star_table$click_bool==1 & star_table$prop_starrating==i])/length(star_table$click_bool[star_table$prop_starrating==i])
  }
st_br<-data.frame(st_br)




#click rates per review score

sqnc<-c(0.5,1,1.5,2,2.5,3,3.5,4,4.5,5)
review_br<-matrix(,10,2)
for(i in seq(1:10)){
  review_br[i,1]<-length(star_table$booking_bool[star_table$booking_bool==1& star_table$click_bool==1 & star_table$prop_review_score==sqnc[i]])/length(star_table$click_bool[star_table$click_bool==1&star_table$prop_review_score==sqnc[i]])
  review_br[i,2]<-length(star_table$click_bool[ star_table$click_bool==1 & star_table$prop_review_score==sqnc[i]])/length(star_table$click_bool[star_table$prop_review_score==sqnc[i]])
}
review_br<-data.frame(review_br)
review_br[1,1]<-0
review_br[1,2]<-0
barplot(t(review_br), col=c("green","blue"),xlab="Review Scores",ylim=c(0,1),names.arg=c(0.5,1,1.5,2,2.5,3,3.5,4,4.5,5))
#click rates per review score
hist(factor_data$prop_location_score2)
#assign prop location score in to bins
location2_buckets<-c(0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1)
location2_Bins<-.bincode(factor_data$prop_location_score2,location2_buckets,TRUE,TRUE)
location2_Bins<-data.frame(location2_Bins)
#click book rates per location2 score bucket
ls2_br<-matrix(,10,2)
location2_table<-cbind(integer_data$click_bool,integer_data$booking_bool,location2_Bins)
names(location2_table)[1]<-paste("click_bool")
names(location2_table)[2]<-paste("booking_bool")
for (i in (1:length(sort(unique(location2_Bins[,]))))){
  ls2_br[i,1]<-length(location2_table$booking_bool[location2_table$booking_bool==1& location2_table$click_bool==1 & location2_table$location2_Bins==i])/length(location2_table$click_bool[location2_table$click_bool==1& location2_table$location2_Bins==i])
  ls2_br[i,2]<-length(location2_table$click_bool[ location2_table$click_bool==1 & location2_table$location2_Bins==i])/length(location2_table$click_bool[location2_table$location2_Bins==i])
}

ls2_br<-data.frame(ls2_br)
names(ls2_br)[1]<-paste("booking rate per location score bucket")
names(ls2_br)[2]<-paste("non booking rate per location score bucket")




par(mar = rep(2, 2))
barplot(t(pr_cbr), col=c("green","blue"),xlab="USD_Price Buckets",names.arg=c(1,2,3,4,5,6,7,8,9))
barplot(t(st_br), col=c("green","blue"),xlab="Star Ratings",names.arg=c(1,2,3,4,5))
barplot(t(review_br), col=c("green","blue"),xlab="Prop. Review Scores",names.arg=c(0.5,1,1.5,2,2.5,3,3.5,4,4.5,5))
barplot(t(ls2_br), col=c("green","blue"),xlab="Location Score 2 Buckets",names.arg=c(0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1))
#Collect variables for CORRELOGRAM
corrVariables<-cbind(factor_data$prop_location_score1,factor_data$prop_location_score2,factor_data$prop_log_historical_price,
                     factor_data$price_usd,integer_data$prop_starrating,integer_data$prop_brand_bool,integer_data$position,
                     integer_data$promotion_flag,integer_data$srch_length_of_stay)
