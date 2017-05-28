##############################
#Clean Console and Environment
##############################
cat("\014") 
rm(list = ls())
##############################
# read cleaned data and put them all together
##############################
WD<-setwd("C:\\Users\\Nikitas Marios\\Desktop\\Data Mining Techniques\\Assignment_2")
load("dataTEST_Pclass1.RData")
load("dataTEST_Pclass2.RData")
load("dataTEST_Pclass3.RData")
load("dataTEST_Pclass4.RData")
load("dataTEST_Pclass5.RData")
load("dataTEST_Pclass6.RData")
##############################
# change the names of the test data frames!!! They have the same sames as the training data frames.
##############################
TestData_PC1<-Data_Pclass1
TestData_PC2<-Data_Pclass2
TestData_PC3<-Data_Pclass3
TestData_PC4<-Data_Pclass4
TestData_PC5<-Data_Pclass5
TestData_PC6<-Data_Pclass6
remove(Data_Pclass1)
remove(Data_Pclass2)
remove(Data_Pclass3)
remove(Data_Pclass4)
remove(Data_Pclass5)
remove(Data_Pclass6)
##############################
# Join all test data
##############################
cd_test<-rbind(TestData_PC1,TestData_PC2,TestData_PC3,TestData_PC4,TestData_PC5,TestData_PC6)
removals<-c("date_time","site_id","visitor_location_country_id","prop_country_id","srch_destination_id","srch_length_of_stay","srch_booking_window",
            "srch_adults_count","srch_children_count","srch_room_count","srch_saturday_night_bool")
cd_test<-cd_test[,!(colnames(cd_test) %in% removals)]
##############################
# Fix the name of the normalized price
##############################
norm_pr_usd<-cd_test$prices_normalized
names(norm_pr_usd)[1]<-paste("norm_price_usd")
cd_test <-cd_test[c(-12)]
cd_test<-cbind(cd_test,norm_pr_usd)
save(cd_test,file="CleanedTestData.RData")
