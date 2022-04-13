### Load data #####
german_credit=read.table("http://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data")
colnames(german_credit)=c("chk_acct", "duration", "credit_his", "purpose", "amount", "saving_acct", "present_emp", "installment_rate", "sex", "other_debtor", "present_resid", "property", "age", "other_install", "housing", "n_credits", "job", "n_people", "telephone", "foreign", "response")
# orginal response coding 1= good, 2 = bad we need 0 = good, 1 = bad
german_credit$response=german_credit$response-1

mean(german_credit$response)
library(dplyr) 

summary(german_credit)
#### Organize the data ######
german_credit$sex <- as.factor(german_credit$sex)
german_credit$chk_acct <- as.factor(german_credit$chk_acct)
german_credit$credit_his <- as.factor(german_credit$credit_his)
german_credit$purpose <- as.factor(german_credit$purpose)
german_credit$saving_acct <- as.factor(german_credit$saving_acct)
german_credit$present_emp <- as.factor(german_credit$present_emp)
german_credit$other_debtor <- as.factor(german_credit$other_debtor)
german_credit$property <- as.factor(german_credit$property)
german_credit$other_install <- as.factor(german_credit$other_install)
german_credit$housing <- as.factor(german_credit$housing)
german_credit$job <- as.factor(german_credit$job)
german_credit$telephone <- as.factor(german_credit$telephone)
german_credit$foreign <- as.factor(german_credit$foreign)


##### Organize the data ######

#(bankruptcy.data$FYEAR <- as.factor(bankruptcy.data$FYEAR)
#bankruptcy.data$DLRSN <- as.factor(bankruptcy.data$DLRSN)
#bankruptcy.data$CUSIP <- as.factor(bankruptcy.data$CUSIP)
#bankruptcy.data$R1 <- as.factor(bankruptcy.data$R1)
#bankruptcy.data$R2 <- as.factor(bankruptcy.data$R2)
#bankruptcy.data$R3 <- as.factor(bankruptcy.data$R3)
#bankruptcy.data$R4 <- as.factor(bankruptcy.data$R4)
#bankruptcy.data$R5 <- as.factor(bankruptcy.data$R5)
#bankruptcy.data$R6 <- as.factor(bankruptcy.data$R6)
#bankruptcy.data$R7 <- as.factor(bankruptcy.data$R7)
#bankruptcy.data$R8 <- as.factor(bankruptcy.data$R8)
#bankruptcy.data$R9 <- as.factor(bankruptcy.data$R9)
#bankruptcy.data$R10 <- as.factor(bankruptcy.data$R10

#### Split data into training and testing data ######

index <- sample(nrow(german_credit),nrow(german_credit)*0.80)
credit_train = german_credit[index,] 
credit_test = german_credit[-index,]

#exploratory data analysis
boxplot(credit_train)
summary(credit_train)

### Model Fitting ####
credit_glm0 <- glm(response~., family=binomial, data=credit_train) 
summary(credit_glm0)

### Check the model's performance ###
credit_glm0$deviance
credit_glm0$dev/credit_glm0$df.residual #mean residual deviance
AIC(credit_glm0)
BIC(credit_glm0)

#prediction performance
hist(predict(credit_glm0))
pred_resp <- predict(credit_glm0,type="response") 
hist(pred_resp)

#missclassification matrix
table(credit_train$response, (pred_resp>0.5)*1, dnn=c("Truth","Predicted"))

table(credit_train$response, (pred_resp>0.2)*1, dnn=c("Truth","Predicted"))

table(credit_train$response, (pred_resp>0.54)*1, dnn=c("Truth","Predicted"))

################ In-sample Fitting ################
pred_glm0_train <- predict(credit_glm0, type="response") 
table(credit_train$response, (pred_glm0_train >0.54)*1, dnn=c("Truth","Predicted"))

################ Asymmetric Cost ################
#Symmetric cost
cost1 <-function(r, pi, pcut){ 
  mean(((r==0)&(pi>pcut)) | ((r==1)&(pi<pcut))) }
#Asymmetric cost
cost2 <-function(r, pi, pcut){ 
  weight1 <-5
  weight0 <-1
  c1 <- (r==1)&(pi<pcut)
  #logical vector - true if actual 1 but predict 0
  c0 <-(r==0)&(pi>pcut)
  #logical vector - true if actual 0 but predict 1
  return(mean(weight1*c1+weight0*c0)) }

pcut <-1/(5+1)
#Symmetric cost
cost1(r = credit_train$response, pi = pred_glm0_train, pcut)

#Asymmetric cost
cost2(r = credit_train$response, pi = pred_glm0_train, pcut)

#ROC Curve
install.packages('ROCR')
library(ROCR) 
################ In-sample Fitting ################
pred <- prediction(pred_glm0_train, credit_train$response) 
perf <- performance(pred,"tpr","fpr") 
plot(perf, colorize=TRUE)
#AUC
unlist(slot(performance(pred,"auc"),"y.values"))

################ Out-sample Prediction ################
pred_glm0_test<- predict(credit_glm0, newdata = credit_test, type= "response")

pred <- prediction(pred_glm0_test, credit_test$response) 
perf <- performance(pred,"tpr","fpr")
plot(perf, colorize=TRUE)
unlist(slot(performance(pred, "auc"),"y.values"))

#### Selection with AIC ######
credit_glm_back <- step(credit_glm0)
# backward selection (if you don't specify anything)
summary(credit_glm_back) 
credit_glm_back$deviance 
AIC(credit_glm_back) 
BIC(credit_glm_back)

#### Selection with BIC ######
credit_glm_back_BIC <- step(credit_glm0, k=log(nrow(credit_train))) 
summary(credit_glm_back_BIC) 
credit_glm_back_BIC$deviance 
AIC(credit_glm_back_BIC) 
BIC(credit_glm_back_BIC)

# Selection with LASSO #
install.packages('glmnet')
library(glmnet)
install.packages("dplyr")
library(dplyr)
colnames(german_credit) = c("chk_acct", "duration", "credit_his",
                            "purpose", "amount", "saving_acct", "present_emp", "installment_rate",
                            "sex", "other_debtor", "present_resid", "property", "age",
                            "other_install", "housing", "n_credits", "job", "n_people",
                            "telephone", "foreign", "response")
dummy <- model.matrix(~ ., data = german_credit)
head(dummy)
german_credit_lasso <- data.frame(dummy[,-1])
index <- sample(nrow(german_credit), nrow(german_credit)*0.80)
german_credit_train_X = as.matrix(select(german_credit_lasso, -response)[index,]) 
german_credit_test_X = as.matrix(select(german_credit_lasso, -response)[-index,]) 
german_credit_train_Y = german_credit_lasso[index,"response"] 
german_credit_test_Y = german_credit_lasso[-index,"response"]
german_credit_lasso = glmnet(x=german_credit_train_X, y=german_credit_train_Y, family='binomial')
plot(german_credit_lasso,label=TRUE)
summary(german_credit_lasso$lambda)
coef(german_credit_lasso,s = 0.001336)
german_crdit_lasso_SelectedVariable = glm(response~ other_install + housing + n_credits + job + n_people + telephone+ foreign, family = binomial, data = credit_train)
AIC(german_crdit_lasso_SelectedVariable)
BIC(german_crdit_lasso_SelectedVariable)
summary(german_credit_lasso$lambda)
pred_german_credit_lasso_train = predict(german_credit_lasso, newx = german_credit_train_X, s = 0.001336, type = 'response')
library(ROCR)
pred = prediction(pred_german_credit_lasso_train, credit_train$response)
perf = performance(pred, "tpr", "fpr")
plot(perf, colorize = TRUE)
unlist(slot(performance(pred,"auc"),"y.values"))
table(credit_train$response, (pred_german_credit_lasso_train > 0.5)*1, dnn = c("Truth","Predicted"))
summary(german_credit_lasso$lambda)
pred_german_credit_lasso_test = predict(german_credit_lasso, newx = credit_test_X, s = 0.001336, type = 'response')
pred = prediction(pred_german_credit_lasso_test, credit_test$response)
perf = performance(pred, "tpr", "fpr")
plot(perf, colorize = TRUE)
unlist(slot(performance(pred,"auc"),"y.values"))
table(credit_test$response, (pred_german_credit_lasso_test > 0.5)*1, dnn = c("Truth","Predicted"))

#Crossvalidation
costfunc <-function(obs, pred.p){ 
  weight1 <-5# define the weight for "true=1 but pred=0" (FN)
  weight0 <-1# define the weight for "true=0 but pred=1" (FP)
  pcut <-1/(1+weight1/weight0) 
  c1 <- (obs==1)&(pred.p < pcut)# count for "true=1 but pred=0" (FN)
  c0 <- (obs==0)&(pred.p >= pcut)# count for "true=0 but pred=1" (FP)
  cost <- mean(weight1*c1 + weight0*c0)
  # misclassification with weight
  return(cost)# you have to return to a value when you write R functions
}

## Reload the data #### 
german_credit1=read.table("http://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data")
library(dplyr) 
german_credit1$sex <- as.factor(german_credit1$sex)
german_credit1$chk_acct <- as.factor(german_credit1$chk_acct)
german_credit1$credit_his <- as.factor(german_credit1$credit_his)
german_credit1$purpose <- as.factor(german_credit1$purpose)
german_credit1$saving_acct <- as.factor(german_credit1$saving_acct)
german_credit1$present_emp <- as.factor(german_credit1$present_emp)
german_credit1$other_debtor <- as.factor(german_credit1$other_debtor)
german_credit1$property <- as.factor(german_credit1$property)
german_credit1$other_install <- as.factor(german_credit1$other_install)
german_credit1$housing <- as.factor(german_credit1$housing)
german_credit1$job <- as.factor(german_credit1$job)
german_credit1$telephone <- as.factor(german_credit1$telephone)
german_credit1$foreign <- as.factor(german_credit1$foreign)
## Cross- Validation ###
library(boot) 
credit_glm1<- glm(response~. , family=binomial, data=german_credit1); 
cv_result <- cv.glm(data=german_credit1, glmfit=credit_glm1, cost=costfunc, K=5) 
cv_result$delta[2]



