library(data.table)
library(ggplot2)
library(caret)
library(gbm)
library(PRROC)
final <- data.frame(fread(file='new_df_10.csv'))
final$property_id <- as.factor(final$property_id) # 151,384 properties
final$category <- as.factor(final$category)       #       6 categories
final$county <- as.factor(final$county)           #       4 counties
final$city <- as.factor(final$city)               #      67 cities
final$zipcode <- as.factor(final$zipcode)         #     175 zipcodes
property_ids <- final$property_id
zipcodes <- final$zipcode
final$property_id <- NULL
final$year <- NULL
# comment if want to keep constant features
final$county <- NULL
final$city <- NULL
final$zipcode <- NULL
final$nrooms <- NULL
final$nbedrooms <- NULL
final$nbaths <- NULL
final$sqft <- NULL

set.seed(1)
train_set <- final[final$target_year %in% c(2010,2011,2012,2013,2014,2015,2016,2017),]
train_set <- train_set[train_set$category=="Hospitality",]
train_pos <- train_set[train_set$sold_t.1==1,]
train_neg <- train_set[train_set$sold_t.1==0,]
train_neg_rows_sample <- sample(rownames(train_neg), nrow(train_pos))
train_neg <- train_set[train_neg_rows_sample,]
train_set <- rbind(train_neg, train_pos)

validate_set <- final[final$target_year %in% c(2018),]
validate_set <- validate_set[validate_set$category=="Hospitality",]

test_set <- final[final$target_year %in% c(2019),]
test_set <- test_set[test_set$category=="Hospitality",]

train_set$target_year <- NULL
validate_set$target_year <- NULL
test_set$target_year <- NULL
train_set$category <- NULL
validate_set$category <- NULL
test_set$category <- NULL
#rm(final)

ntrees=1000
boost <- gbm(`sold_t.1`~., data=train_set, distribution="bernoulli", n.trees=ntrees, 
                 interaction.depth=6,
                 shrinkage=0.01,)
summary(boost)

# choosing best ntrees based on ROC AUC
rocs_validate <- c()
for (ntree in 1:ntrees) {
  validate_pred <- predict(boost, newdata=validate_set, n.trees=ntree, type="response")
  fg <- validate_pred[validate_set$sold_t.1 == 1]
  bg <- validate_pred[validate_set$sold_t.1 == 0]
  roc_validate <- roc.curve(scores.class0 = fg, scores.class1 = bg, curve = F)
  rocs_validate <- c(rocs_validate, roc_validate$auc)
  
  #train_pred <- predict(boost, n.trees=ntree, type="response")
  #fg <- train_pred[train_set$sold_t.1 == 1]
  #bg <- train_pred[train_set$sold_t.1 == 0]
  #roc_train <- roc.curve(scores.class0 = fg, scores.class1 = bg, curve = F)
  #rocs_train <- c(rocs_train, roc_train$auc)
}

# best ntrees <= ROC AUC
plot(rocs_validate, type="l", col="lightcoral", xlab="Number of Trees", ylab="ROC AUC",
     main="Retail", lwd=2,
     )
#lines(rocs_train)
best_ntree <- which.max(rocs_validate)

# train ROC/PR
train_pred <- predict(boost,n.trees=best_ntree, type="response")
plot(train_pred)
fg <- train_pred[train_set$sold_t.1 == 1]
bg <- train_pred[train_set$sold_t.1 == 0]
roc <- roc.curve(scores.class0 = fg, scores.class1 = bg, curve = T)
plot(roc)
#
pr <- pr.curve(scores.class0 = fg, scores.class1 = bg, curve = T)
plot(pr)
# train score distribution
true_scores <- data.frame(score=train_pred[train_set$sold_t.1==1])
false_scores <- data.frame(score=train_pred[train_set$sold_t.1==0])
true_scores$label <- "sold"
false_scores$label <- "not sold"
scores <- rbind(true_scores, false_scores)
ggplot(scores, aes(score, fill = label)) + geom_density(alpha = 0.2)
#
ggplot(scores, aes(score, fill = label)) + 
  geom_histogram(alpha = 0.5, aes(y = ..density..), position = 'identity')

# validation ROC/PR
validate_pred <- predict(boost, newdata=validate_set, n.trees=best_ntree, type="response")
fg <- validate_pred[validate_set$sold_t.1 == 1]
bg <- validate_pred[validate_set$sold_t.1 == 0]
roc <- roc.curve(scores.class0 = fg, scores.class1 = bg, curve = T)
plot(roc)
#
pr <- pr.curve(scores.class0 = fg, scores.class1 = bg, curve = T)
plot(pr)
# validation score distribution
true_scores <- data.frame(score=validate_pred[validate_set$sold_t.1==1])
false_scores <- data.frame(score=validate_pred[validate_set$sold_t.1==0])
true_scores$label <- "sold"
false_scores$label <- "not sold"
scores <- rbind(true_scores, false_scores)
ggplot(scores, aes(score, fill = label)) + geom_density(alpha = 0.2)  + ggtitle("Retail") +  theme(plot.title = element_text(hjust = 0.5))
#
ggplot(scores, aes(score, fill = label)) + 
  geom_histogram(alpha = 0.5, aes(y = ..density..), position = 'identity')

# best threshold
thresholds <- c()
tns <- c()
fns <- c()
fps <- c()
tps <- c()
validate_set$sold_t.1 <- as.factor(validate_set$sold_t.1)
for (threshold in seq(0,1,length=100)) {
  thresholds <- c(thresholds, threshold)
  validate_pred_class <- factor(ifelse(validate_pred>threshold, 1, 0), levels=c(0,1))
  tble <- table(validate_set$sold_t.1, validate_pred_class)
  tn <- tble[1]
  fn <- tble[2]
  fp <- tble[3]
  tp <- tble[4]
  tns <- c(tns, tn)
  fns <- c(fns, fn)
  fps <- c(fps, fp)
  tps <- c(tps, tp)
}
recalls <- tps/(tps+fns)
precisions <- tps/(tps+fps)
f1scores <- 2*recalls*precisions/(precisions+recalls)
plot(thresholds, f1scores)
plot(thresholds, precisions)
plot(thresholds, recalls)
precisions[which.max(f1scores)]
recalls[which.max(f1scores)]
# best_threshold = thresholds[which.max(f1scores)]
best_threshold = 0.36
# retrain train+validation
full_train <- rbind(train_set, validate_set)
boost <- gbm(`sold_t.1`~., data=full_train, distribution="bernoulli", n.trees=best_ntree, 
             interaction.depth=6,
             shrinkage=0.01,)
# test
test_pred <- predict(boost, newdata=test_set, n.trees=best_ntree, type="response")
test_set$sold_t.1 <- as.factor(test_set$sold_t.1)
test_pred_class <- factor(ifelse(test_pred>best_threshold, 1, 0), levels=c(0,1))
tble <- table(test_set$sold_t.1, test_pred_class)
## test ROC/PR
plot(test_pred)
fg <- test_pred[test_set$sold_t.1 == 1]
bg <- test_pred[test_set$sold_t.1 == 0]
roc <- roc.curve(scores.class0 = fg, scores.class1 = bg, curve = T)
plot(roc)
#
pr <- pr.curve(scores.class0 = fg, scores.class1 = bg, curve = T)
plot(pr)
# test score distribution
true_scores <- data.frame(score=test_pred[test_set$sold_t.1==1])
false_scores <- data.frame(score=test_pred[test_set$sold_t.1==0])
true_scores$label <- "sold"
false_scores$label <- "not sold"
scores <- rbind(true_scores, false_scores)
ggplot(scores, aes(score, fill = label)) + geom_density(alpha = 0.2)
ggplot(scores, aes(score, fill = label)) + 
  geom_histogram(alpha = 0.5, aes(y = ..density..), position = 'identity')
#
thresholds <- c()
tns <- c()
fns <- c()
fps <- c()
tps <- c()
test_set$sold_t.1 <- as.factor(test_set$sold_t.1)
for (threshold in seq(0,1,length=100)) {
  thresholds <- c(thresholds, threshold)
  test_pred_class <- factor(ifelse(test_pred>threshold, 1, 0), levels=c(0,1))
  tble <- table(test_set$sold_t.1, test_pred_class)
  tn <- tble[1]
  fn <- tble[2]
  fp <- tble[3]
  tp <- tble[4]
  tns <- c(tns, tn)
  fns <- c(fns, fn)
  fps <- c(fps, fp)
  tps <- c(tps, tp)
}
recalls <- tps/(tps+fns)
precisions <- tps/(tps+fps)
f1scores <- 2*recalls*precisions/(precisions+recalls)
plot(thresholds, f1scores)
plot(thresholds, precisions)
plot(thresholds, recalls)
precisions[which.max(f1scores)]
recalls[which.max(f1scores)]

# join on property ids for map
x <- data.frame(true=test_set$sold_t.1, pred=test_pred_class, prob=test_pred, row.names=rownames(test_set))
y <- data.frame(property_id=property_ids, zipcode=zipcodes, row.names=1:length(property_ids))
z <- merge(x,y,by="row.names", all.x=TRUE)
rownames(z) <- z$Row.names
z$Row.names <- NULL
write.csv(z, "hospitality.csv", row.names=FALSE)