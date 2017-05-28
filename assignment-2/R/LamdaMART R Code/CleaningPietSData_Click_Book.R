##############################
#Clean Console and Environment
##############################
cat("\014") 
rm(list = ls())
##############################
# read cleaned data and put them all together
##############################
WD<-setwd("C:\\Users\\Nikitas Marios\\Desktop\\Data Mining Techniques\\Assignment_2")
load("data_Pclass1.RData")
load("data_Pclass2.RData")
load("data_Pclass3.RData")
load("data_Pclass4.RData")
load("data_Pclass5.RData")
load("data_Pclass6.RData")
##############################
# adjust column names because they were mistaken
##############################
varnames <- c("srch_id","date_time","site_id","visitor_location_country_id","prop_country_id","prop_id","prop_starrating","prop_review_score","prop_brand_bool","prop_location_score1","prop_location_score2",
                 "prop_log_historical_price","position","price_usd","promotion_flag","srch_destination_id","srch_length_of_stay","srch_booking_window",
                 "srch_adults_count","srch_children_count","srch_room_count","srch_saturday_night_bool","random_bool","click_bool","booking_bool","year","month","consumer","normalized_prices","Pclass", "score")
names(Data_Pclass1)<-varnames
names(Data_Pclass2)<-varnames
names(Data_Pclass3)<-varnames
names(Data_Pclass4)<-varnames
names(Data_Pclass5)<-varnames
names(Data_Pclass6)<-varnames
##############################
# change the mistaken last column's name and remove column/varaible 30
##############################
#Data_Pclass1<-Data_Pclass1[c(-30)]
#Data_Pclass2<-Data_Pclass2[c(-30)]
#Data_Pclass3<-Data_Pclass3[c(-30)]
#Data_Pclass4<-Data_Pclass4[c(-30)]
#Data_Pclass5<-Data_Pclass5[c(-30)]
#Data_Pclass6<-Data_Pclass6[c(-30)]
##############################
# Sort persearch id and within each one of them you sort based on the score. This means that
# for each search id number the top row of it would be the one with the highest score number
##############################
Data_Pclass1<- Data_Pclass1[order(Data_Pclass1$srch_id, -Data_Pclass1[,c(31)]),]
Data_Pclass2<- Data_Pclass2[order(Data_Pclass2$srch_id, -Data_Pclass2[,c(31)]),]
Data_Pclass3<- Data_Pclass3[order(Data_Pclass3$srch_id, -Data_Pclass3[,c(31)]),]
Data_Pclass4<- Data_Pclass4[order(Data_Pclass4$srch_id, -Data_Pclass4[,c(31)]),]
Data_Pclass5<- Data_Pclass5[order(Data_Pclass5$srch_id, -Data_Pclass5[,c(31)]),]
Data_Pclass6<- Data_Pclass6[order(Data_Pclass6$srch_id, -Data_Pclass6[,c(31)]),]
##############################
# keep separately the click and the booking variables for each data set and remove from data_pclass along with dates and months
##############################
cb1<-Data_Pclass1[c(24,25)]
cb2<-Data_Pclass2[c(24,25)]
cb3<-Data_Pclass3[c(24,25)]
cb4<-Data_Pclass4[c(24,25)]
cb5<-Data_Pclass5[c(24,25)]
cb6<-Data_Pclass6[c(24,25)]
Data_Pclass1<-Data_Pclass1[c(-2,-26,-27)]
Data_Pclass2<-Data_Pclass2[c(-2,-26,-27)]
Data_Pclass3<-Data_Pclass3[c(-2,-26,-27)]
Data_Pclass4<-Data_Pclass4[c(-2,-26,-27)]
Data_Pclass5<-Data_Pclass5[c(-2,-26,-27)]
Data_Pclass6<-Data_Pclass6[c(-2,-26,-27)]
##############################
# Form up final cleaned dataset
##############################
cd<-rbind(Data_Pclass1,Data_Pclass2,Data_Pclass3,Data_Pclass4,Data_Pclass5,Data_Pclass6)
removals<-c("site_id","visitor_location_country_id","prop_country_id","prop_id","position","srch_destination_id","srch_length_of_stay","srch_booking_window",
            "srch_adults_count","srch_children_count","srch_room_count","srch_saturday_night_bool")
cd<-cd[,!(colnames(cd) %in% removals)]
norm_pr_usd<-cd$normalized_prices
names(norm_pr_usd)[1]<-paste("norm_price_usd")
cd<-cd[c(-14)]
cd<-cbind(cd,norm_pr_usd)
save(cd,file="CleanedData_Click_Book.RData")
