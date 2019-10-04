

Setting the seed to '12345' for rerunning and loading all required libraries.

```{r}
set.seed(12345)

library(leaps)
library(MASS)
library(ROCR)
library(ggplot2)
library(e1071) 
```


```{r}
df <- read.csv("Fundraising.csv",header = T)
test.df<-read.csv("FutureFundraising.csv",header = T)
dim(test.df)
head(df)
## STEP #0: Deleting unnecessary columns from our data.

df$Row.Id  <- NULL
df$Row.Id. <- NULL
df$TARGET_D <- NULL
```

The "Fundraising.csv" dataset was partitioned into 60% training and 40% validation sets. We deleted the Row Id, Row Id., and TARGET_D columns from the dataset.
In this particular case, responders to the fundraising drive is the rarer class and more important than the non-responders. Such an asymmetric response and cost requires oversampling the donor class to obtain more useful data for improving the performance of the classifiers.
Since the response rate is 5.1%, a simple random sampling from the original dataset would have yielded too few relevant classes to build a strong predictive model. Therefore, a stratified sampling with a disproportionate weighting of the donor class is used for the training data.
We have used logistic regression to build our first model.

```{r}
## STEP #1: Partitioning our dataset. (60% Training, 40% Validation).

train.rows <- sample(rownames(df), dim(df)[1]*0.6)
train.df <- df[train.rows, ]

valid.rows <- setdiff(rownames(df), train.rows) 
valid.df <- df[valid.rows, ]

```
```{r}
# Full Model.
full.model <- glm(TARGET_B ~ ., data = train.df, family = "binomial")
summary(full.model)

# Model after AIC method.
step.model <- stepAIC(full.model, direction = "both", trace = TRUE)
summary(step.model)

# Reduced model (after dropping "zipconvert_2", "zipconvert_3", "zipconvert_4", "zipconvert_5" due to extremely high p-values).
reduced.model <- glm(TARGET_B ~ NUMCHLD + INCOME + NUMPROM + LASTGIFT + totalmonths, data = train.df, family = "binomial")
summary(reduced.model)

# Further reduced model/final model (after dropping "NUMPROM" due to p-value > 10%).
final.model <- glm(TARGET_B ~ NUMCHLD + INCOME + LASTGIFT + totalmonths, data = train.df, family = "binomial")
summary(final.model)

reduced.df<-df[,c(6,7,17)]
head(reduced.df)
```

A logistic regression analysis was conducted using all the predictor variables initially on the training set. The model was further refined by using stepwise regression to eliminate predictors that were not significant. The predictors for our final model include NUMCHLD, INCOME, LASTGIFT, and totalmonths.
We used Excel to calculate the optimum cutoff value for the model. A value of 0.498 resulted in the maximum profit and minimum cost. Therefore 0.5 was set as the cutoff for classifying the entries in the model.
The accuracy of the model obtained from the validation data is 50.32%

```{r}
predict_valid <- predict(final.model, valid.df, type = "response")
table(valid.df$TARGET_B,predict_valid>0.5)
```
```{r}
#Profit from the validation data
Profit_valid<-(358*((13-0.68)/9.8))
Profit_valid
#cost from the validation data
cost_valid<-(272*((0-0.68)/0.53))
cost_valid
```

Net profit for donors = $13 - $0.68 = $12.32 Net profit for non-donors = -$0.68 Adjusted net profit for donors = $12.32 / 9.38 = $1.3134 Adjusted net profit for non-donors = -$0.68 / 0.53 = -$1.283
For the first model, net profit for validation data = $450.0571 - $348.9811 = $101.076

```{r}
predict_train <- predict(final.model, train.df, type = "response")
table(train.df$TARGET_B,predict_train>0.5)
```
```{r}
#Profit from the training data
Profit_train<-(533*((13-0.68)/9.8))
Profit_train
#cost from the training data
cost_train<-(397*((0-0.68)/0.53))
cost_train
```

Net profit for the training data = $670.0571 - $509.3585 = $160.6986

```{r}
#ROCR Curve

ROCRpred <- prediction(predict_train, train.df$TARGET_B)
ROCRperf <- performance(ROCRpred, 'tpr','fpr')
plot(ROCRperf, colorize = TRUE, text.adj = c(-0.2,1.7))
```

The above Roc plot is for the training dataset

```{r}
ROCRpred <- prediction(predict_valid, valid.df$TARGET_B)
ROCRperf <- performance(ROCRpred, 'tpr','fpr')
plot(ROCRperf, colorize = TRUE, text.adj = c(-0.2,1.7))
```

The above ROC plot is for the validation dataset. The ROC curve gives a better indication of the model performance in cases where asymmetric costs are involved by plotting the sensitivity over specificity. The closer the curve gets to the top-left corner, the better is the performance. In this case, the ROC curve signifies a slightly better performance than the naive rule which would have yielded a diagonal.
The model was run on the FutureFundraising dataset to classify the entries.

```{r}
predict_test <- predict(final.model, test.df, type = "response")
test_TARGET_B <- ifelse(predict_test>0.5,1,0)
head(test_TARGET_B)
```

The probabilities of the entries in FutureFundraising dataset has been sorted in descending order.

```{r}
pred_test <- predict(final.model, newdata = test.df, type = "response") 
fund.pred <- pred_test[order(-pred_test)] 
head(fund.pred) 
```


SVM

Here,We used SVM to build our second model. The svm() function was used to fit the model to the training data.

```{r}
library(e1071) 
```
```{r}

df <- read.csv("Fundraising.csv",header = T)
head(df)
test.df<-read.csv("FutureFundraising.csv",header = T)
## STEP #0:Removing unnecessary columns from our dataset.

df$Row.Id  <- NULL
df$Row.Id. <- NULL
df$TARGET_D <- NULL

# Encoding the target feature as factor 
df$TARGET_B = factor(df$TARGET_B, levels = c(0, 1)) 
```


```{r}
## STEP #1: Partitioning our data. (60% Training, 40% Validation).

train.rows <- sample(rownames(df), dim(df)[1]*0.6)
train.df <- df[train.rows, ]

valid.rows <- setdiff(rownames(df), train.rows) 
valid.df <- df[valid.rows, ]

```

```{r}
## STEP #2: Model Building.

attach(train.df)
attach(test.df)

# Fitting SVM to the Training set 


classifier = svm(formula = TARGET_B ~ .,  data = train.df, type = 'C-classification', kernel = 'linear') 
classifier

```

```{r}
# Predicting the Test set results for validation data
y_pred_valid = predict(classifier, newdata = valid.df) 
y_pred_valid
# Making the Confusion Matrix for validation data
cm_valid = table(valid.df$TARGET_B, y_pred_valid) 
cm_valid
```

```{r}
#Profit from the validation data
Profit_valid<-(362*((13-0.68)/9.8))
Profit_valid
#cost from the validation data
cost_valid<-(253*((0-0.68)/0.53))
cost_valid
```

```{r}
# Predicting the Test set results for training data
y_pred_train = predict(classifier, newdata = train.df) 
y_pred_train


# Making the Confusion Matrix for training data
cm_train = table(train.df$TARGET_B, y_pred_train) 
cm_train

```

The accuracy of the model obtained from the validation data is 54.73%.

```{r}
#Profit from the training data
Profit_valid<-(572*((13-0.68)/9.8))
Profit_valid
#cost from the training data
cost_valid<-(373*((0-0.68)/0.53))
cost_valid

```

For the second model, net profit for validation data = $455.0857 - $324.6038 = $130.4819

```{r}
pred_test_svm <- predict(classifier, newdata = test.df, type = "response") 
fund.pred_svm <- pred_test[order(-pred_test)] 
head(fund.pred_svm) 
```

From the results obtained, we can conclude that the SVM model is superior than the logistic regression model. Since SVM provides class predictions, we can use the probabilities obtained from the logistic regression model to predict the donors and non-donors. We will use a cut-off value of 0.6 for predicting the donors. Therefore, all the entries having a probability greater than 0.6 will be considered for the mailing campaign.
