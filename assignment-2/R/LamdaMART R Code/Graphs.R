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
##############################
#CALCULATE GRAPH VARIABLES
##############################
#per P class
PClass<-matrix(,6,2)
for (i in (1:6)){
  PClass[i,1]<-length(cd$booking_bool[cd$booking_bool==1& cd$click_bool==1 & cd$Pclass==i])/length(cd$click_bool[cd$click_bool==1& cd$Pclass==i])
  PClass[i,2]<-length(cd$click_bool[ cd$click_bool==1 & cd$Pclass==i])/length(cd$click_bool[cd$Pclass==i])
}
#per C class
CClass<-matrix(,4,2)
for (i in (1:4)){
  CClass[i,1]<-length(cd$booking_bool[cd$booking_bool==1& cd$click_bool==1 & cd$consumer==i])/length(cd$click_bool[cd$click_bool==1& cd$consumer==i])
  CClass[i,2]<-length(cd$click_bool[ cd$click_bool==1 & cd$consumer==i])/length(cd$click_bool[cd$consumer==i])
}
par(mfrow=c(1,2))
barplot(t(PClass), col=c("green","blue"),xlab="People Class",legend = colnames(PClass),names.arg=c(1,2,3,4,5,6),ylim = c(0,1))
barplot(t(CClass), col=c("green","blue"),xlab="Consumer Class",legend = colnames(CClass),names.arg=c("Single","Couple","Parents","Other"),ylim = c(0,1))

load("GraphRates.RData")
par(mfrow = c(2, 2))
barplot(t(pr_cbr), col=c("green","blue"),xlab="USD_Price Buckets",ylim=c(0,1),names.arg=c(1,2,3,4,5,6,7,8,9))
barplot(t(st_br), col=c("green","blue"),xlab="Star Ratings",ylim=c(0,1),names.arg=c(1,2,3,4,5))
barplot(t(review_br), col=c("green","blue"),xlab="Prop. Review Scores",ylim=c(0,1),names.arg=c(0.5,1,1.5,2,2.5,3,3.5,4,4.5,5))
barplot(t(ls2_br), col=c("green","blue"),xlab="Location Score 2 Buckets",ylim=c(0,1),names.arg=c(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1))
