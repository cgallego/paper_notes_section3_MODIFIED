## AUC comparison: Boostrap method
# AUC difference significance testing
#   
#   * Computation details:
#   a) boot.n (2000) bootstrap replicates are drawn from the data. If boot.stratified = TRUE, each replicate contains exactly the same number of controls and cases than the original sample
#   
#   b) for each bootstrap replicate, the AUC of the two ROC curves are computed and the difference is used:
#   $$  D =  \frac{AUC_1-AUC_2}{std(AUC_1-AUC_2)} \sim Z$$
#     where $std$ is the standard deviation of the bootstrap differences and AUC1 and AUC2 the AUC of the two (bootstrap replicate) ROC curves.
#   
#   c) Z approximately follows a normal distribution, one or two-tailed p-values can be calculated accordingly.

## ROC Comparison: Ensembles of trees
# group1: Tr_RF, ciobj_Tr_RF, val_RF, ciobj_val_RF, Te_RF
# group 2: Tr_RF_nxG, ciobj_Tr_RF_nxG, val_RF_nxG, ciobj_val_RF_nxG, Te_RF_nxG
# group 3: Tr_DECMLP, ciobj_Tr_DECMLP, val_DECMLP, ciobj_val_DECMLP, Te_DECMLP
################
library(pROC)
library(readr)
setwd("Z:/Cristina/Section3/paper_notes_section3_MODIFIED")
exp1_pooled_pred_train <- read_csv("datasets/exp1_pooled_pred_train.csv")
exp1_pooled_pred_val <- read_csv("datasets/exp1_pooled_pred_val.csv")
exp1_pred_test <- read_csv("datasets/exp1_pred_test.csv")

# Create 5 objects: Tr_RF, ciobj_Tr_RF, val_RF, ciobj_val_RF, Te_RF
# plot ROC curve
################
library(pROC)
par(mfrow = c(1, 3))
par(cex = 0.6)
par(mar = c(3, 3, 0, 0), oma = c(0.51, 0.51, 0.51, 0.51))
n=12
colors = rainbow(n, s = 1, v = 1, start = 0, end = max(1, n - 1)/n, alpha = 1)
# plot
Tr_exp1 <- plot.roc(exp1_pooled_pred_train$labels, exp1_pooled_pred_train$probC, col=colors[1], lty=1)
ciobj_Tr_exp1 <- ci.se(Tr_exp1, specificities=seq(0, 1, 0.05)) 
plot(ciobj_Tr_exp1, type="shape", col="grey") # plot as a grey shape
par(new=TRUE)
plot.roc(exp1_pooled_pred_train$labels, exp1_pooled_pred_train$probC, col=colors[1], lty=1, main="ROC for cvTrain")
legend("bottomright", 
       legend = c(paste0("cvTrain: AUC=", format(Tr_exp1$auc,digits=3, format="f"))), 
       col = colors[1],lwd = 2, lty = c(1))

# plot
Val_exp1 <- plot.roc(exp1_pooled_pred_val$labels, exp1_pooled_pred_val$probC, col=colors[1], lty=1)
ciobj_val_exp1 <- ci.se(Val_exp1, specificities=seq(0, 1, 0.05)) 
plot(ciobj_val_exp1, type="shape", col="grey") # plot as a blue shape
par(new=TRUE)
plot.roc(exp1_pooled_pred_val$labels, exp1_pooled_pred_val$probC, col=colors[10], lty=1, main="ROC for cvValidation")
legend("bottomright", 
       legend = c(paste0("cvValidation: AUC=", format(Val_exp1$auc, digits=3, format="f"))), 
       col = colors[10],lwd = 2, lty = c(1))
# plot
Te_exp1 <- plot.roc(exp1_pred_test$labels, exp1_pred_test$probC, col=colors[8], lty=1)
ciobj_Te_exp1 <- ci.se(Te_exp1, specificities=seq(0, 1, 0.05)) 
plot(ciobj_Te_exp1, type="shape", col="grey") # plot as a grey shape
par(new=TRUE)
plot.roc(exp1_pred_test$labels, exp1_pred_test$probC, col=colors[8], lty=1, main="ROC for held-out Test")
legend("bottomright", 
       legend = c(paste0("Test: AUC=", formatC(Te_exp1$auc,digits=2, format="f"))), 
       col = colors[8],lwd = 2, lty = c(1))


# Do we significantly overtrain?
roc.test(Tr_exp1, Val_exp1, method="bootstrap",boot.stratified=TRUE, alternative="greater")

roc.test(Tr_exp1, Te_exp1, method="bootstrap",boot.stratified=TRUE, alternative="greater")


