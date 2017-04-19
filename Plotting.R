# In this file the data of the titanic is analyzed.
# Furthermore mining algortihm's are implemented to learn a
# predictive analysis, about who survived.

# Code to save figures is commented :)



rm(list=ls())        # Delete workspace

# Read data
train <- read.csv("C:/Users/Piet/Documents/Titanic/train.csv", header= TRUE)
test <- read.csv("C:/Users/Piet/Documents/Titanic/test.csv", header = TRUE)

str(train)

# Load packages
library(ggplot2)
library(stringr)
library(Hmisc)


# Add a "Survived" variable to the test set to allow for combining data sets
# Source: https://github.com/EasyD/IntroToDataScience/blob/master/TitanicDataAnalysis_Video7.R

test1 <- data.frame(Survived = rep("None", nrow(test)), test[,])   # Add in a clumsy way a Survive variable to the test dataset
test= test1
rm(test1)

# Transforms variables:
 
train$Sex <- as.numeric(train$Sex == "male")              # Make gender a numeric variable
test$Sex <- as.numeric(test$Sex == "male")

train$Age <- as.factor(train$Age)                         # Factorize the variable age for the training data et
test$Age <- as.factor(test$Age)                         # Factorize the variable age,for the testing data


test$Fare[is.na(test$Fare)] <- median(test$Fare, na.rm=TRUE) # Replace single missing value of Fare with it's median


# Cabin

extractTitle <- function(Cabin) {Cabin <- as.character(Cabin) # Read only the letter's out of each element in Cabin

if (length(grep("A", Cabin)) > 0) {   
  return ("A")
} else if (length(grep("B", Cabin)) > 0) {
  return ("B")
} else if (length(grep("C", Cabin)) > 0) {
  return ("C")
} else if (length(grep("D", Cabin)) > 0) {
  return ("D")
} else if (length(grep("E", Cabin)) > 0) {
  return ("E")
} else if (length(grep("F", Cabin)) > 0) {
  return ("F")
} else {
  return ("Unknown")
}
}

Cabins  <- NULL                                        # Replace the variable Cabin with only it's letter
for (i in 1:nrow(train)) {
  Cabins <- c(Cabins, extractTitle(train[i,"Cabin"]))  }
 train$Cabin <- as.factor(Cabins)                           # Factorize the variable cabin

 Cabinstest <- NULL 
 for (i in 1:nrow(test)) {
   Cabinstest <- c(Cabinstest, extractTitle(test[i,"Cabin"]))  }
 test$Cabin <- as.factor(Cabinstest)         

# Name 
extractTitle <- function(Name) {Name <- as.character(Name) # Read only the titles out of each element in Cabin

if (length(grep("Miss.", Name)) > 0) {
  return ("Miss.")
} else if (length(grep("Master.", Name)) > 0) { 
  return ("Master.")
} else if (length(grep("Mrs.", Name)) > 0) {
  return ("Mrs.")
} else if (length(grep("Mr.", Name)) > 0) {
  return ("Mr.")
} else {
  return ("Other")
}
}

titles <- NULL
for (i in 1:nrow(train)) {                                   # Make a new variable: title
  titles <- c(titles, extractTitle(train[i,"Name"]))  }
train$title <-titles                                         # Factorize the variable Title

titlestest <- NULL
for (i in 1:nrow(test)) {                                   # Make a new variable: title
  titlestest <- c(titlestest, extractTitle(test[i,"Name"]))  }
test$title <-titlestest                                         # Factorize the variable Title



# Deal with missing values

library(rpart)


Agefit <- rpart(Age ~ Pclass + Sex + SibSp + Parch + Fare + Embarked + title,   # Set up model to Predict Missing values for Age
data = train[!is.na(train$Age), ], method = "anova")

train$Age[is.na(train$Age)] <- round(predict(Agefit, train[is.na(train$Age), ]))  # Predict Age


Agefit2 <- rpart(Age ~ Pclass + Sex + SibSp + Parch + Fare + Embarked + title,   # Set up model to Predict Missing values for Age
                    data = test[!is.na(test$Age), ], method = "anova")

test$Age1[is.na(test$Age)] <- round(predict(Agefit2, test[is.na(test$Age), ]))   # Predict Age



test$Age1[is.na(test$Age1)] = test$Age[is.na(test$Age1)]      # Replace Age variable with the complete Age variable
test$Age <- test$Age1

train <- train[ , -which(names(train) %in% c("Name","Ticket"))]   # Delete some variables
test <- test[ , -which(names(test) %in% c("Name","Ticket","Age1"))]


rm(Cabins, i, titles, extractTitle, Cabinstest, titlestest, Agefit,Agefit2)                 # Delete some values
str(train)



# Give distributions with barplot's
# b1 <- barplot(prop.table(table(train$Survived)),
# xlab=c("Male","Female"))
# # jpeg("b_Survived.jpg")
# b2 <- barplot(prop.table(table(train$Pclass)))
# # jpeg("b_Pclass.jpg")
# b3 <- barplot(prop.table(table(train$Sex)))
# # jpeg("b_Sex.jpg")
# b4 <- barplot(prop.table(table(train$Age)))
# # jpeg("b_Age.jpg")
# b5 <- barplot(prop.table(table(train$SibSp)))
# # jpeg("b_SibSp.jpg")
# b6 <- barplot(prop.table(table(train$Parch)))
# # jpeg("b_Parch.jpg")
# b7 <- barplot(prop.table(table(train$Fare)))
# # jpeg("b_Fare.jpg")
# b8 <- barplot(prop.table(table(train$Cabin)))
# # jpeg("b_Cabin.jpg")
# b9 <- barplot(prop.table(table(train$Embarked)))
# jpeg("b_Embarked.jpg")
# # b10 <- barplot(prop.table(table(train$title)))



# Plot Pie's:

library(mapplots)

# par(mfrow=c(1,2))
# p1 <- pie(table((train$Sex[train$Survived==1])),main="Sex which Survived", labels = c("Female","Male"))
# p1 <- pie(table((train$Sex[train$Survived==0])),main="Sex which did not Survive", labels = c("Female","Male"))
# # jpeg("pie_Sex.jpg")     # Save Pie
# 
# par(mfrow=c(1,2))
# pie(table((train$Pclass[train$Survived==1])),main="Pclass which Survived",labels = c("Upper","Middel","Lower"))
# pie(table((train$Pclass[train$Survived==0])),main="Pclass which did not Survive",labels = c("Upper","Middel","Lower"))
# # jpeg("pie_Pclass.jpg")
# 
# par(mfrow=c(1,2))
# pie(table((train$Age[train$Survived==1])),main="Age which Survived",labels = c("0-5","5-18","18-40","40-60","60-80","80-120"))
# pie(table((train$Age[train$Survived==0])),main="Age which did not Survive",labels = c("0-5","5-18","18-40","40-60","60-80","80-120"))
# # jpeg("pie_Age.jpg")
# 
# par(mfrow=c(1,2))
# pie(table((train$Parch[train$Survived==1])),main="Parch which Survived",labels = c("0","1","2","3","4","5","6","7"))
# pie(table((train$Parch[train$Survived==0])),main="Parch which did not Survive",labels = c("0","1","2","3","4","5","6","7"))
# # jpeg("pie_Parch.jpg")
# 
# par(mfrow=c(1,2))
# pie(table((train$Cabin[train$Survived==1])),main="Cabin which Survived",labels = c("A","B","C","D","E","F","Unknown"))
# pie(table((train$Cabin[train$Survived==0])),main="Cabin which did not Survive",labels = c("A","B","C","D","E","F","Unknown"))
# # jpeg("pie_Cabin.jpg")
# 
# par(mfrow=c(1,2))
# pie(table((train$title[train$Survived==1])),main="Titles which Survived",labels = c("Master","Miss.","Mr.","Mrs.","Other"))
# pie(table((train$title[train$Survived==0])),main="Titles which did not Survive",labels = c("Master","Miss.","Mr.","Mrs.","Other"))
# # jpeg("pie_title.jpg")

# Plot's
cabindata <- train[order(train[,9],decreasing=TRUE),]    # Construct a new dataset to analyze the variable Cabin


# ggplot(train[1:891,], aes(x = Sex, fill = factor(Survived))) +
#   stat_count(width = 0.5) +
#   facet_wrap(~Pclass) + 
#   ggtitle("Pclass") +
#     xlab("Sex: Female - Male") +
#   ylab("Total Count") +
#   labs(fill = "Survived")
#   
# plot_Pclass_title <- ggplot(train[1:891,], aes(x = Age, fill = factor(Survived))) +
#     stat_count(width = 0.5) +
#     facet_wrap(~Pclass) + 
#     ggtitle("Pclass") +
#     xlab("Title") +
#     ylab("Total Count") +
#     labs(fill = "Survived")
# # jpeg("plot_Pclass_title.jpg")
#   
#   plot_Pclasss_Age <- ggplot(train[1:891,], aes(x = title, fill = factor(Survived))) +
#     stat_count(width = 0.5) +
#     facet_wrap(~Pclass) + 
#     ggtitle("Pclass") +
#     xlab("Age") +
#     ylab("Total Count") +
#     labs(fill = "Survived")
#   # jpeg("plot_Pclasss_Age.jpg")
#   
#   ggplot(cabindata[1:692,], aes(x = title, fill = factor(Survived))) +
#     stat_count(width = 0.5) +
#     facet_wrap(~Cabin) + 
#     ggtitle("Cabin") +
#     xlab("Title") +
#     ylab("Total Count") +
#     labs(fill = "Survived")
#   
#   plot_title_Cabin <- ggplot(cabindata[693:891,], aes(x = title, fill = factor(Survived))) +
#     stat_count(width = 0.5) +
#     facet_wrap(~Cabin) +
#     ggtitle("Cabin") +
#     xlab("Title") +
#     ylab("Total Count") +
#     labs(fill = "Survived")
#   # jpeg("plot_title_Cabin")
  
  # Load files
  library(Hmisc)
  library(polycor)
  library('corrplot')  
  
  # Make a correlation Matrix       [UNSOLVED]
  CorrMatrix <-rcorr(as.matrix(train[,1:6]), type="spearman")
  CorrMatrix
  

  
  # Strange transformation [DELETE THIS!]
 
  # factorize data:
  train$Survived <- as.factor(train$Survived)
  train$Pclass <- as.factor(train$Pclass)
  train$Sex <- as.factor(train$Sex)
  train$title <- as.factor(train$title)
   
  test$Survived <- as.factor(test$Survived)
  test$Pclass <- as.factor(test$Pclass)
  test$Sex <- as.factor(test$Sex)
  test$title <- as.factor(test$title)
  
  train$Fare <- as.numeric(train$Fare)
  train$SibSp <- as.numeric(train$SibSp)
  train$Age <- as.numeric(train$Age)
  train$Parch <- as.numeric(train$Parch)
  
  test$SibSp <- as.numeric(test$SibSp)
  test$Fare <- as.numeric(test$Fare)
  test$Age <- as.numeric(test$Age)
  test$Parch <- as.numeric(test$Parch)
  


  # Data Mining
  
  # Logistic Regression
  # Source: https://rpubs.com/Vincent/Titanic
  library(glmnet)
  
 
  extractDat <- function(data){               # Extract specific data
    Data <- c('Pclass',
                  'Sex',
                  'Age',
                  'SibSp',
                  'Parch',
                  'Fare',
                  'Cabin',
                  'Embarked',
                  'Survived',
                  'title'
                  
    )
    da <- data[ , Data]
    return(da)
  }
  
  # Create an item matrix
  a <- model.matrix(Survived~., data = extractDat(train))
  # Take the survived vector
  b <- extractDat(train)$Survived
  
  
  # Make an item matrix for Training and Testing data set
   item2<- model.matrix(~., data = extractDat(train)[,-which(names(extractDat(train)) %in% 'Survived')])
   item2test<- model.matrix(~., data = extractDat(test)[,-which(names(extractDat(test)) %in% 'Survived')])
                         
                         
  set.seed(1)
  
  # Make binominal model
  mlogr <- cv.glmnet(a, b, alpha = 0, family = 'binomial', type.measure = 'deviance')
  
  logrpredic <- predict(mlogr , newx = item2, s = 'lambda.min', type='class')
  
   OutputLRegT <- data.frame(PassengerId = train$PassengerId, Survived =logrpredic)
   aLR <- table(OutputLRegT[ , 2][OutputLRegT[ , 2] %in% train$Survived])    # Estimate number of matches
   aLR
   PrecisionLogRegr <- aLR[1]/nrow(train)      # Estimate precision training data
   PrecisionLogRegr
   
  
   
   
   
   
   
   
   
   
   
   
   
  
  
  # Association Rules:
  
  # Group age variable
  train$Age <-as.factor(findInterval(train$Age, c(0,20, 40,60,80)))
  test$Age <-as.factor(findInterval(test$Age, c(0,20, 40,60,80)))
  
  # Group Fare variable
  train$Fare <-as.factor(findInterval(train$Fare, c(0,10, 20,60,80,200)))
  test$Fare <-as.factor(findInterval(test$Fare, c(0,10, 20,60,80,200)))
  

  

# Factorize vectors
train$SibSp <- as.factor(train$SibSp)
train$Parch <- as.factor(train$Parch)
train$Fare <- as.factor(train$Fare)

test$SibSp <- as.factor(test$SibSp)
test$Parch <- as.factor(test$Parch)
test$Fare <- as.factor(test$Fare)
  
  
  # load package
  library(arules)
  library(arulesViz)
  
  
  # Mine for Association rules
  rules <- apriori(train[ , 2:10], control= list(verbose=F) ,parameter= list(minlen=2, supp=0.05, conf=0.900), appearance = list(rhs=c("Survived=0","Survived=1"), default ="lhs"))
  quality(rules) <- round(quality(rules),  digits = 3)
  
  
  rules.sorted= sort(rules, by="confidence")
  inspect(rules.sorted)
  
  # Recognize redundant rules
  M.subset <- is.subset(rules.sorted, rules.sorted)
  redundant <- is.redundant(rules.sorted)
  which(redundant)
  
  # Delete redundant rules
  rules.pruned <- rules.sorted[!redundant]
  inspect(rules.pruned)

  plot(rules.pruned)               # Plot leftover rules
  plot(rules.pruned, method="matrix3D", measure="confidence", control = list(reorder=TRUE));
  

  


    
  
  # Item Matrices

  LHS <- rules.pruned@lhs
  
  # Make an item matrix for Training and Testing data set

  item2<- model.matrix(~., data = extractDat(train)[,-which(names(extractDat(train)) %in% 'Survived')])
  item2test<- model.matrix(~., data = extractDat(test)[,-which(names(extractDat(test)) %in% 'Survived')])
  
  # dimnames(m) <- list(NULL, paste("item", c(1:20), sep=""))
  
  itemMatrixtrain <- as(item2, "itemMatrix")                        # Create item matrix
  itemMatrixtest <- as(item2test, "itemMatrix")
  
 
  rulesMatchLHS_train <- is.subset(rules.pruned@lhs, itemMatrixtrain)     # Match item matrices with LHS rules
  rulesMatchLHS_test <- is.subset(rules.pruned@lhs, itemMatrixtest)


  # Match new variable with is NOT a subset of the current basket (so that some items are left as potential recommendation)
  suitableRules_train <-  rulesMatchLHS_train & !(is.subset(rules.pruned@rhs,itemMatrixtrain))
  suitableRules_test <-  rulesMatchLHS_test & !(is.subset(rules.pruned@rhs,itemMatrixtest))
  
  
  
  
  # here they are
  inspect(rules.pruned[suitableRules_train])
  

  
  classes_train <- predict(train$Survived, factor(x))
  classes_test <- predict(test$Survived, factor(x))
  
  
  
  
  # now extract the matching rhs ...
  recommendations <- strsplit(LIST(rules.pruned@rhs)[[1]],split=" ")
  recommendations <- lapply(recommendations,function(x){paste(x,collapse=" ")})
  recommendations <- as.character(recommendations)
  
  # ... and remove all items which are already in the basket
  recommendations <- recommendations[!sapply(recommendations,function(x){basket %in% x})]
  
  print(recommendations)
  
  
  
  
  
  
  
  rulesMatch <- is.subset(rules.pruned@lhs,test)
  
  # find all rules whose lhs matches the training example
  rulesMatch <- is.subset(rules.pruned@lhs,train[1])
  
  # subset all applicable rules
  applicable <- rules.pruned[rulesMatch==TRUE]
  
  # the first rule has the highest confidence since they are sorted
  prediction <- applicable[1]
  inspect(prediction@rhs)
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  # Random forest
  # Source: https://www.kaggle.com/benhamner/titanic/random-forest-benchmark-r/code
  
  library('randomForest')
  
  # Read data again
  trainRF <- read.csv("C:/Users/Piet/Documents/Titanic/train.csv", header= TRUE, stringsAsFactors=FALSE)
  testRF <- read.csv("C:/Users/Piet/Documents/Titanic/test.csv", header = TRUE, stringsAsFactors=FALSE)
  
  # Fill in previous variables
  trainRF$Name  <- as.character(train$title)
  testRF$Name  <- as.character(test$title)
  trainRF$Cabin  <- as.character(train$Cabin)
  testRF$Cabin  <- as.character(test$Cabin)
  trainRF$Age  <- (train$Age)
  testRF$Age  <- (test$Age)
  testRF$Fare  <- (test$Fare)
  
  extractData <- function(data) {   # Extract data
    Dat <- c("Pclass",
                  "Age",
                  "Sex",
                  "Parch",
                  "Name",
                  "SibSp",
                  "Fare",
                  "Embarked")
    da <- data[,Dat]
    da$Embarked[da$Embarked==""] = "S"
    da$Name     <- as.factor(da$Name)
    da$Embarked <- as.factor(da$Embarked)
    da$Sex      <- as.factor(da$Sex)
    return(da)
  }
  
  # Construct a random Forest
  Forest <- randomForest(extractData(trainRF), as.factor(trainRF$Survived), ntree=100, importance=TRUE)
  
  
  outputT <- data.frame(PassengerId = trainRF$PassengerId)               # Use RF method to estimate Survivors
  outputT$Survived <- predict(Forest, extractData(trainRF))
  
  a <- table(outputT$Survived[outputT$Survived %in% trainRF$Survived])    # Estimate number of matches
  PrecisionRandomForest <- a[1]/nrow(trainRF)
  PrecisionRandomForest
  

  hierarchie <- importance(Forest, type=1)                # Plot importance results
  hierarchietable <- data.frame(Feature=row.names(hierarchie), Importance=hierarchie[,1])
  plotRandomForest <- ggplot(hierarchietable, aes(x=reorder(Feature, Importance), y=Importance)) +
       coord_flip() +
    geom_bar(stat="identity", fill="#53cfff") +
    theme_light(base_size=20) +
    xlab("") +
    ylab("Importance") + 
    ggtitle("Order of importance of the variables") 
    
  plotRandomForest

  
  rm(rules.sorted,logrpredic, cabindata,a,b,aLR,CorrMatrix,PrecisionLogRegr,redundant,mlogr)
