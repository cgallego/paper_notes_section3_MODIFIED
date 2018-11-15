library(pROC)
library(readr)
library(plyr)
library(dplyr)
library(caret)



## Recursive Feature Elimination 
setwd("Z:/Cristina/Section3/paper_notes_section3_MODIFIED")
allNMEs_dynamic <- read_csv("datasets/dyn_roi_records_allNMEs_descStats.csv")
allNMEs_morphology <- read_csv("datasets/morpho_roi_records_allNMEs_descStats.csv")
allNMEs_texture <- read_csv("datasets/text_roi_records_allNMEs_descStats.csv")
allNMEs_stage1 <- read_csv("datasets/stage1_roi_records_allNMEs_descStats.csv")

discrall_dict_allNMEs =  read_csv("datasets/named_nxGnormfeatures_allNMEs_descStats.csv")
YnxG_allNME  =  read_csv('datasets/YnxG_allNME.csv')

# combine all
combX_allNME <- cbind(allNMEs_dynamic %>%
  select(-X1),
allNMEs_morphology %>%
  select(-X1),
allNMEs_texture %>%
  select(-X1),
allNMEs_stage1 %>%
  select(-X1),
discrall_dict_allNMEs %>%
  select(-X1))


# The predictors are centered and scaled:
normalization <- preProcess(combX_allNME)
print(normalization)

normcombX_allNME <- predict(normalization, combX_allNME)
normcombX_allNME <- as.data.frame(normcombX_allNME)
y <- YnxG_allNME$classNME

## ,There are a number of pre-defined sets of functions for several models, including: linear regression (in the object lmFuncs), random forests (rfFuncs), naive Bayes (nbFuncs), bagged trees (treebagFuncs) and functions
subsets <- c(c(10, 25, 50, 75), (16:30)*5, c(200, 250, 300, 350, 400, 450, 500))
set.seed(10)
ctrl <- rfeControl(functions = rfFuncs,
                   method = "repeatedcv",
                   repeats = 5,
                   verbose = FALSE)

cleaned_data <- na.omit(cbind.data.frame(normcombX_allNME,y))

# ommit the unknown class
labeledy <- cleaned_data %>%
  filter(y != "U") %>%
  select(y)
labeledX <- cleaned_data %>%
  filter(y != "U") %>%
  select(-y)
y <- factor(labeledy$y)


lmProfile <- rfe(labeledX, y,
                 sizes = subsets,
                 rfeControl = ctrl)

lmProfile

# The lmProfile is a list of class "rfe" that contains an object fit that is the final linear model with the remaining terms. The model can be used to get predictions for future or test samples.
predictors(lmProfile)
lmProfile$fit

head(lmProfile$resample)

## Best subset size
trellis.par.set(caretTheme())
plot(lmProfile, type = c("g", "o"))
