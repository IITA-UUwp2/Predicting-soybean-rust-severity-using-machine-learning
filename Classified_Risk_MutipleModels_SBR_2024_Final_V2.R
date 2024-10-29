### A workflow for predicting categorical aflatoxin risk classes in South and East Africa
### Various ensemble and non-ensemble machine learning methods are used
### VSURF was used to eliminate redundant variables
### The model uses both space-time (ST-CV) and random (CV)
### Created by Stella M Gachoki
### June/July 2024

##### ********************* INITIALIZE ******************** ####
#### Create a function to load all the libraries required
load_libraries <- function() {
  libs <- c("caret", "randomForest", "sf", "CAST", "gbm", "xgboost", "tidyquant",
            "VSURF", "glmnet", "reshape2", "ggplot2", "dplyr", "tidyr", 
            "corrplot", "igraph", "themis", "MLmetrics", "adabag", "C50", 
            "tidyquant", "viridis", "gridExtra", "stringr", "raster", 
            "terra", "stars", "rasterVis", "sp", "virtualspecies","ggdist","BiodiversityR", "pdp","Boruta")
  sapply(libs, require, character.only = TRUE)
}

load_libraries()

#### Set the working directory
setwd("C:/Users/gstel/OneDrive/Desktop/IITA/Soybean Rust/Project/Soybean Rust severity")

# Load the sbrorally matched database
d.sbr <- read.csv("dbase_comp_SBR_combined_Final_V1.csv")
names(d.sbr)

### Make the relevant columns categorical
d.sbr$majstage <- as.factor(d.sbr$majstage)
d.sbr$majVar <- as.factor(d.sbr$majVar)
d.sbr$Year <- as.factor(d.sbr$Year)
d.sbr$Country <- as.factor(d.sbr$Country)
d.sbr$class <- as.factor(d.sbr$class)

### Explarotary data analysis. Plot the class against variety and flowering stage.
colors <- c("c1" = "grey","c2" = "springgreen","c3" = "violet","c4" = "orange","c5" = "red1")

p.var <- ggplot(d.sbr, aes(x = majVar, fill = class)) +geom_bar(position = "dodge") + theme_ggdist()+
  labs(title = "Soybean variety", x = "", y = "Count")+ scale_fill_manual(values = colors) + 
  theme(legend.position = "none", text = element_text(size = 14, color = "black"), 
        axis.title = element_text(size = 14, color = "black"), 
        plot.title = element_text(size = 18, hjust = 0.5, face = "bold"),
        axis.text.x = element_text(size = 14, color = "black",angle = 45, hjust = 1),
        axis.text.y = element_text(size = 14, color = "black"))

p.stage <- ggplot(d.sbr, aes(x = majstage, fill = class)) + geom_bar(position = "dodge") + theme_ggdist()+
  labs(title = "Growth stage", x = "", y = "")+ scale_fill_manual(values = colors, labels = c("0", "1 to 10%", "11 to 35%", "36-65%", ">65%")) + 
  theme(legend.position = "none", text = element_text(size = 14, color = "black"), legend.text = element_text(),
        legend.key.height = unit(2, 'cm'), legend.key.width = unit(1, 'cm'), 
        legend.background = element_rect(fill = "snow"),
        axis.title = element_text(size = 14, color = "black"), 
        plot.title = element_text(size = 18, hjust = 0.5, face = "bold"),
        axis.text.x = element_text(size = 14, color = "black",angle = 45, hjust = 1),
        axis.text.y = element_text(size = 14, color = "black"))

png(file = "Variety_Rstage_SBR.png", width = 12000, height = 9000, units = "px", res = 700, type = "cairo")
grid.arrange(p.var,p.stage,nrow=2)
dev.off()

######## ********************** MODEL FITTING USING REPEATED CV ***************** ####
##### Data partition using space-time partitioning ####
set.seed(123)  # For reproducibility
# Create stratified sampling indices
trainIndex <- createDataPartition(d.sbr$class, p = 0.8, list = FALSE, times = 1)
trainData.sbr <- d.sbr[trainIndex, ]
testData.sbr <- d.sbr[-trainIndex, ]

# Verify the split
print(dim(trainData.sbr))
print(dim(testData.sbr))
print(table(trainData.sbr$class))
print(table(testData.sbr$class))

# Split data into predictors (X) and response variable (y): dynamic predictor only
x.sbr <- trainData.sbr[, c(20:73)]
y.sbr <- trainData.sbr$class

##### Feature elimination Methods usings VSURF (strict) and BORUTA (less strict) #####
set.seed(123)
##BORUTA
boruta.sbr <- Boruta(x = x.sbr, y = y.sbr, doTrace = 2, maxRuns = 100)
print(boruta.sbr)
boruta_features <- getSelectedAttributes(boruta.sbr, withTentative = FALSE)
print(boruta_features)

# Define the formula for model fitting
fm.boruta <- class ~  aviJunJul+eviJunJul+ndmiJunJul+tmaxJunJul+tminJunJul+windJunJul+precJunJul+pdsiJunJul+maxdewJunJul+
  mindewJunJul+etiJunJul+dsr12JunJul+dsr15JunJul+eviNovDec+gciNovDec+gndviNovDec+ndmiNovDec+tmaxNovDec+  
  tminNovDec+precNovDec+smoistNovDec+pdsiNovDec+vpdNovDec+mindewNovDec+etiNovDec+eviDecJan+gciDecJan+   
  gndviDecJan+ndmiDecJan+tmaxDecJan+tminDecJan+windDecJan+precDecJan+smoistDecJan+pdsiDecJan+vpdDecJan+   
  maxdewDecJan+mindewDecJan+etiDecJan+dsr09DecJan+dsr12DecJan+dsr15DecJan 

##Define the training control parameters. Use repeated CV.
ctrl.sbr <- trainControl(method = "repeatedcv", number=5, repeats=3,savePredictions = "all", verboseIter = TRUE,
                         classProbs = TRUE, summaryFunction = multiClassSummary,selectionFunction = "best",
                         allowParallel = TRUE)

## Compare four modelling techniques using BORUTA retained vars and VSURF vars
### BORUTA 
models.boruta <- list(
  ranger.boruta = caret::train(fm.boruta, data = trainData.sbr, method = "ranger", trControl = ctrl.sbr,importance="permutation",
                            tuneGrid = expand.grid(.mtry =10, .splitrule = "extratrees", .min.node.size = 5), 
                            num.trees = 300,max.depth=6,min.bucket=1),
  xgbTree.boruta = caret::train(fm.boruta, data = trainData.sbr, method = "xgbTree", trControl = ctrl.sbr,tuneGrid = expand.grid(nrounds = 10, 
                                  max_depth = 6, eta =  0.05, min_child_weight = 5,subsample = 0.8, gamma = 1,colsample_bytree =  0.7)),
  gbm.boruta = caret::train(fm.boruta, data = trainData.sbr, method = "gbm", trControl = ctrl.sbr,tuneGrid = expand.grid(n.trees = 500, 
                                                interaction.depth = 6, shrinkage =0.03, n.minobsinnode = 10),verbose=FALSE)
)

### Training metrics for temporally matched with Repeated CV
results.boruta <- resamples(models.boruta)
summary(results.boruta)
metrics.boruta <- as.data.frame(results.boruta$values)
metrics_long.boruta <- metrics.boruta %>% pivot_longer(cols = -Resample, names_to = "Model", values_to = "Value") %>%
  separate(Model, into = c("Model", "Metric"), sep = "~")
as.data.frame(metrics_long.boruta)
metrics_long.boruta$Model <- str_extract(metrics_long.boruta$Model, "[^.]+")

accuracy.boruta <- metrics_long.boruta %>% filter(Metric == "Accuracy")
av_acc.boruta <- metrics_long.boruta %>%filter(Metric == "Accuracy") %>%group_by(Model) %>%
  summarise(Avg_accuracy = median(Value, na.rm = TRUE))

bal_accuracy.boruta <- metrics_long.boruta %>% filter(Metric == "Mean_Balanced_Accuracy")
bal_av_acc.boruta <- metrics_long.boruta %>%filter(Metric == "Mean_Balanced_Accuracy") %>%group_by(Model) %>%
  summarise(bal_Avg_accuracy = median(Value, na.rm = TRUE))

auc.boruta <- metrics_long.boruta %>% filter(Metric == "AUC")
auc_av_acc.boruta <- metrics_long.boruta %>%filter(Metric == "AUC") %>%group_by(Model) %>%
  summarise(auc_Avg_accuracy = median(Value, na.rm = TRUE))

f1.boruta <- metrics_long.boruta %>% filter(Metric == "Mean_F1")
f1_av_acc.boruta <- metrics_long.boruta %>%filter(Metric == "Mean_F1") %>%group_by(Model) %>%
  summarise(f1_Avg_accuracy = median(Value, na.rm = TRUE))

acc.boruta.plt <- ggplot(accuracy.boruta, aes(x = Model, y = Value)) +
  stat_halfeye(adjust = 0.5, justification = -0.2, .width = 0, point_colour = NA) +
  geom_boxplot(fill = "lightblue", color = "darkblue",width = 0.2, outlier.color = NA, alpha = NA) +
  stat_dots(side = "left", justification = 1, binwidth = NA, dotsize = 0.1, overlaps = "nudge") +
  theme_bw() +geom_text(data = av_acc.boruta, aes(x = Model, y = Avg_accuracy, label = round(Avg_accuracy, 2)),
                        vjust = -5, hjust = 0.5, size =6, color = "navy") +
  labs(title = "Overall accuracy", x = "", y = "") + ylim(0,1)+
  theme(legend.position = "none", text = element_text(size = 16, color = "black"), 
        axis.title.y = element_text(size = 16, color = "black", face = "bold"), 
        plot.title = element_text(size = 16, hjust = 0.5, face = "bold"),
        axis.text.x = element_blank(),
        axis.text.y = element_text(size = 16, color = "black"))

bal_acc.boruta.plt <- ggplot(bal_accuracy.boruta, aes(x = Model, y = Value)) +
  stat_halfeye(adjust = 0.5, justification = -0.2, .width = 0, point_colour = NA) +
  geom_boxplot(fill = "lightblue", color = "darkblue",width = 0.2, outlier.color = NA, alpha = NA) +
  stat_dots(side = "left", justification = 1, binwidth = NA, dotsize = 0.1, overlaps = "nudge") +
  theme_bw() +geom_text(data = bal_av_acc.boruta, aes(x = Model, y = bal_Avg_accuracy, label = round(bal_Avg_accuracy, 2)),
                        vjust = -5, hjust = 0.5, size = 6, color = "navy") +
  labs(title = "Balanced accuracy", x = "", y = "") + ylim(0,1)+
  theme(legend.position = "none", text = element_text(size = 16, color = "black"), 
        axis.title.y = element_text(size = 16, color = "black", face = "bold"), 
        plot.title = element_text(size = 16, hjust = 0.5, face = "bold"),
        axis.text.x = element_blank(),
        axis.text.y = element_text(size = 16, color = "black"))

auc_acc.boruta.plt <- ggplot(auc.boruta, aes(x = Model, y = Value)) +
  stat_halfeye(adjust = 0.5, justification = -0.2, .width = 0, point_colour = NA) +
  geom_boxplot(fill = "lightblue", color = "darkblue",width = 0.2, outlier.color = NA, alpha = NA) +
  stat_dots(side = "left", justification = 1, binwidth = NA, dotsize = 0.1, overlaps = "nudge") +
  theme_bw() +geom_text(data = auc_av_acc.boruta, aes(x = Model, y = auc_Avg_accuracy, label = round(auc_Avg_accuracy, 2)),
                        vjust = -2, hjust = 0.2, size =6, color = "navy") +
  labs(title = "Area under curve", x = "", y = "") + ylim(0,1)+
  theme(legend.position = "none", text = element_text(size = 16, color = "black"), 
        axis.title.y = element_text(size = 16, color = "black", face = "bold"), 
        plot.title = element_text(size = 16, hjust = 0.5, face = "bold"),
        axis.text.x = element_text(size = 16, color = "black", hjust = 0.5),
        axis.text.y = element_text(size = 16, color = "black"))

f1_acc.boruta.plt <- ggplot(f1.boruta, aes(x = Model, y = Value)) +
  stat_halfeye(adjust = 0.5, justification = -0.2, .width = 0, point_colour = NA) +
  geom_boxplot(fill = "lightblue", color = "darkblue",width = 0.2, outlier.color = NA, alpha = NA) +
  stat_dots(side = "left", justification = 1, binwidth = NA, dotsize = 0.1, overlaps = "nudge") +
  theme_bw() +geom_text(data = f1_av_acc.boruta, aes(x = Model, y = f1_Avg_accuracy, label = round(f1_Avg_accuracy, 2)),
                        vjust = -5, hjust = 0.5, size =6, color = "navy") +
  labs(title = "F1-score", x = "", y = "") + ylim(0,1)+
  theme(legend.position = "none", text = element_text(size = 16, color = "black"), 
        axis.title.y = element_text(size = 16, color = "black",face = "bold"), 
        plot.title = element_text(size = 16, hjust = 0.5, face = "bold"),
        axis.text.x = element_text(size = 16, color = "black", hjust = 0.5),
        axis.text.y = element_text(size = 16, color = "black"))

#getModelInfo()$svmLinear$parameters

png(file = "Training_Metrics_SBR.png", width = 9000, height =9000, units = "px", res = 800, type = "cairo")
grid.arrange(acc.boruta.plt, bal_acc.boruta.plt,auc_acc.boruta.plt, f1_acc.boruta.plt, ncol=2)
dev.off()

###### TESTING Evaluation metrics
#### Temporarly matched ST-CV
# Initialize an empty list to store predictions
predictions.boruta <- list()

# Loop through each model in the models.boruta list to make predictions
for(model_name in names(models.boruta)) {
  model <- models.boruta[[model_name]]
  predictions.boruta[[model_name]] <- predict(model, newdata = testData.sbr)
}
outcome_variable_name <- "class"  # Change this to your actual outcome variable name

evaluation_results.boruta <- list()
for(model_name in names(predictions.boruta)) {
  prediction <- predictions.boruta[[model_name]]
  actual <- testData.sbr[[outcome_variable_name]]
  evaluation_results.boruta[[model_name]] <- confusionMatrix(prediction, actual)
}

#write.csv(evaluation_results.boruta$xgbTree.boruta$byClass,"xgbTreeboruta.csv")

### Load the test metrics and make plots
test.CV.boruta <- read.csv("TestMetrics_Boruta.csv")

png(file = "Test_Metrics_SBR.png", width = 8000, height =4000, units = "px", res = 850, type = "cairo")
ggplot(test.CV.boruta, aes(x = Metric, y = Class, fill = Value)) +
  geom_tile(color = "white") + facet_wrap(~ Model) +scale_fill_gradient(low = "white", high = "blue") +
  labs(title = "", x = "", y = "") + theme_minimal() +
  theme(legend.position = "none",text = element_text(size = 20, color = "black"),axis.title = element_text(size = 14, color = "black"),
        plot.title = element_text(size = 20, hjust = 0.5, face = "bold"),axis.text.x = element_text(size = 18, color = "black", hjust = 0.5),
        axis.text.y = element_text(size = 20, color = "black")) +
  geom_text(aes(label = sprintf("%.2f", Value)), size = 6, color = "black")
dev.off()

#### Variable importance and partial dependence plots
imp.gbm.boruta <- varImp(models.boruta$gbm.boruta)
imp.gbm.boruta2 <- as.data.frame(imp.gbm.boruta$importance)
imp.gbm.boruta2$varnames <- rownames(imp.gbm.boruta2)
#Extract the top 30 variables
top20_vars <- head(imp.gbm.boruta2[order(imp.gbm.boruta2$Overall, decreasing = TRUE), ], 20)

png(file = "varImp_SBR.png", width = 8000, height =6000, units = "px", res = 750, type = "cairo")
ggplot(top20_vars, aes(x=reorder(varnames, Overall), y=Overall)) +  geom_point(color="blue",size=4)+
  ggtitle("Gradient boosting model")+ xlab("") + ylab("")+ coord_flip()+theme_tq() + 
  theme(plot.title = element_text(size=16, hjust = 0.5, color="black"),
        text = element_text(size = 16, face="bold",color = "black"))
dev.off()

##### partial dependence plots
### Class 2 1 to 10%
#Boruta
pdp.boruta = topPredictors(models.boruta$gbm.boruta,n=9)
pd.boruta.c2 <- NULL
for (i in pdp.boruta) {
  tmp.c2 <- partial(models.boruta$gbm.boruta, pred.var = i, data = trainData.boruta, type = "classification",which.class = 2L)
  names(tmp.c2) <- c("x", "y")
  pd.boruta.c2 <- rbind(pd.boruta.c2, cbind(tmp.c2, predictor = i))
}
pd.boruta.c2$predictor <- factor(pd.boruta.c2$predictor, levels = unique( pd.boruta.c2$predictor))

### Class 3 11 to 35%
#Boruta
pdp.boruta = topPredictors(models.boruta$gbm.boruta,n=9)
pd.boruta.c3 <- NULL
for (i in pdp.boruta) {
  tmp.c3 <- partial(models.boruta$gbm.boruta, pred.var = i, data = trainData.boruta, type = "classification",which.class = 3L)
  names(tmp.c3) <- c("x", "y")
  pd.boruta.c3 <- rbind(pd.boruta.c3, cbind(tmp.c3, predictor = i))
}
pd.boruta.c3$predictor <- factor(pd.boruta.c3$predictor, levels = unique( pd.boruta.c3$predictor))

### Class 4 36 to 65%
#Boruta
pdp.boruta = topPredictors(models.boruta$gbm.boruta,n=9)
pd.boruta.c4 <- NULL
for (i in pdp.boruta) {
  tmp.c4 <- partial(models.boruta$gbm.boruta, pred.var = i, data = trainData.boruta, type = "classification",which.class = 4L)
  names(tmp.c4) <- c("x", "y")
  pd.boruta.c4 <- rbind(pd.boruta.c4, cbind(tmp.c4, predictor = i))
}
pd.boruta.c4$predictor <- factor(pd.boruta.c4$predictor, levels = unique( pd.boruta.c4$predictor))

### Class 5 >65%
#Boruta
pdp.boruta = topPredictors(models.boruta$gbm.boruta,n=9)
pd.boruta.c5 <- NULL
for (i in pdp.boruta) {
  tmp.c5 <- partial(models.boruta$gbm.boruta, pred.var = i, data = trainData.boruta, type = "classification",which.class = 5L)
  names(tmp.c5) <- c("x", "y")
  pd.boruta.c5 <- rbind(pd.boruta.c5, cbind(tmp.c5, predictor = i))
}
pd.boruta.c5$predictor <- factor(pd.boruta.c5$predictor, levels = unique( pd.boruta.c5$predictor))

#### Combine the PDP databases together for more interpretative plotting
pd.boruta.c2 <- pd.boruta.c2 %>% mutate(severity = "c2")
pd.boruta.c3 <- pd.boruta.c3 %>% mutate(severity = "c3")
pd.boruta.c4 <- pd.boruta.c4 %>% mutate(severity = "c4")
pd.boruta.c5 <- pd.boruta.c5 %>% mutate(severity = "c5")
combined_boruta <- bind_rows(pd.boruta.c2, pd.boruta.c3, pd.boruta.c4, pd.boruta.c5)

png(file = "PartialPlots_SBR_Boruta_GBM.png", width = 12000, height =14000, units = "px", res = 800, type = "cairo")
ggplot(combined_boruta, aes(x = x, y = y, color = severity)) + geom_line(linewidth = 1.5) + 
  facet_wrap(~ predictor, scales = "free")+
  ggtitle("Boruta") + xlab("") + ylab("log-odds") + 
  scale_color_manual(values = c("c2" = "springgreen","c3" = "violet","c4" = "orange","c5" = "red1"), 
                     labels = c("1 to 10%", "11 to 35%", "36-65%", ">65%")) + 
  theme_minimal() + theme(legend.position="top",plot.title = element_text(size = 20, hjust = 0.5, color = "black"),
    text = element_text(size = 20, face = "bold", color = "black"),
    axis.title = element_text(size = 16),axis.text = element_text(size = 16))
dev.off()

#### Spatial predictions, future projections and novel conditions calculations.
ras<- rast("Raster_stack_AllSBR.tif")
names(ras)

#2018 - WET
ret.ras18.boruta <- c("AVI_2018_JunJul","EVI_2018_JunJul","NDMI_2018_JunJul","tmax2018JunJul","tmin2018JunJul","wind2018JunJul","prec2018JunJul","pdsi2018JunJul","maxdew2018JunJul",
"mindew2018JunJul","eti2018JunJul","dsr122018JunJul","dsr152018JunJul","EVI_2018_NovDec","GCI_2018_NovDec","GNDVI_2018_NovDec","NDMI_2018_NovDec","tmax2018NovDec","tmin2018NovDec",
"prec2018NovDec","smoist2018NovDec","pdsi2018NovDec","vpd2018NovDec","mindew2018NovDec","eti2018NovDec","EVI_2018_DecJan","GCI_2018_DecJan","GNDVI_2018_DecJan","NDMI_2018_DecJan",
"tmax2018DecJan","tmin2018DecJan","wind2018DecJan","prec2018DecJan","smoist2018DecJan","pdsi2018DecJan","vpd2018DecJan","maxdew2018DecJan","mindew2018DecJan","eti2018DecJan",
"dsr092018DecJan","dsr122018DecJan","dsr152018DecJan" )
gbm.ras.boruta18 <- ras[[ret.ras18.boruta]]
names(gbm.ras.boruta18) <- c("aviJunJul","eviJunJul","ndmiJunJul","tmaxJunJul","tminJunJul","windJunJul","precJunJul","pdsiJunJul","maxdewJunJul",
"mindewJunJul","etiJunJul","dsr12JunJul","dsr15JunJul","eviNovDec","gciNovDec","gndviNovDec","ndmiNovDec","tmaxNovDec","tminNovDec","precNovDec",
"smoistNovDec","pdsiNovDec","vpdNovDec","mindewNovDec","etiNovDec","eviDecJan","gciDecJan","gndviDecJan","ndmiDecJan",
"tmaxDecJan","tminDecJan","windDecJan","precDecJan","smoistDecJan","pdsiDecJan","vpdDecJan","maxdewDecJan","mindewDecJan","etiDecJan","dsr09DecJan","dsr12DecJan","dsr15DecJan" )

pred.gbm.boruta18 <- predict(object=gbm.ras.boruta18,model=models.boruta$gbm.boruta,na.rm=T)
gc()
plot(pred.gbm.boruta18)
writeRaster(pred.gbm.boruta18,"prediction_boruta_wet2018_gbm.tif",overwrite=TRUE)

names.novel.boruta <- c("aviJunJul","eviJunJul","ndmiJunJul","tmaxJunJul","tminJunJul","windJunJul","precJunJul","pdsiJunJul","maxdewJunJul",
                     "mindewJunJul","etiJunJul","dsr12JunJul","dsr15JunJul","eviNovDec","gciNovDec","gndviNovDec","ndmiNovDec","tmaxNovDec","tminNovDec","precNovDec",
                     "smoistNovDec","pdsiNovDec","vpdNovDec","mindewNovDec","etiNovDec","eviDecJan","gciDecJan","gndviDecJan","ndmiDecJan",
                     "tmaxDecJan","tminDecJan","windDecJan","precDecJan","smoistDecJan","pdsiDecJan","vpdDecJan","maxdewDecJan","mindewDecJan","etiDecJan","dsr09DecJan","dsr12DecJan","dsr15DecJan" )
d.novel.boruta.frame <- d.sbr[,c(names.novel.boruta)]
novel.test.boruta <- ensemble.novel.object(d.novel.boruta.frame, name="noveltest")

rastack.novel.boruta18 <- stack(gbm.ras.boruta18)
novel.raster.boruta18 <- ensemble.novel(x=rastack.novel.boruta18, novel.object=novel.test.boruta)
writeRaster(novel.raster.boruta18,"prediction_boruta_wet2018_novel_gbm.tif",overwrite=TRUE)

## 2016 - Dry
ret.ras16.boruta <- c("AVI_2016_JunJul","EVI_2016_JunJul","NDMI_2016_JunJul","tmax2016JunJul","tmin2016JunJul","wind2016JunJul","prec2016JunJul","pdsi2016JunJul","maxdew2016JunJul",
                      "mindew2016JunJul","eti2016JunJul","dsr122016JunJul","dsr152016JunJul","EVI_2016_NovDec","GCI_2016_NovDec","GNDVI_2016_NovDec","NDMI_2016_NovDec","tmax2016NovDec","tmin2016NovDec",
                      "prec2016NovDec","smoist2016NovDec","pdsi2016NovDec","vpd2016NovDec","mindew2016NovDec","eti2016NovDec","EVI_2016_DecJan","GCI_2016_DecJan","GNDVI_2016_DecJan","NDMI_2016_DecJan",
                      "tmax2016DecJan","tmin2016DecJan","wind2016DecJan","prec2016DecJan","smoist2016DecJan","pdsi2016DecJan","vpd2016DecJan","maxdew2016DecJan","mindew2016DecJan","eti2016DecJan",
                      "dsr092016DecJan","dsr122016DecJan","dsr152016DecJan" )
gbm.ras.boruta16 <- ras[[ret.ras16.boruta]]
names(gbm.ras.boruta16) <- c("aviJunJul","eviJunJul","ndmiJunJul","tmaxJunJul","tminJunJul","windJunJul","precJunJul","pdsiJunJul","maxdewJunJul",
                             "mindewJunJul","etiJunJul","dsr12JunJul","dsr15JunJul","eviNovDec","gciNovDec","gndviNovDec","ndmiNovDec","tmaxNovDec","tminNovDec","precNovDec",
                             "smoistNovDec","pdsiNovDec","vpdNovDec","mindewNovDec","etiNovDec","eviDecJan","gciDecJan","gndviDecJan","ndmiDecJan",
                             "tmaxDecJan","tminDecJan","windDecJan","precDecJan","smoistDecJan","pdsiDecJan","vpdDecJan","maxdewDecJan","mindewDecJan","etiDecJan","dsr09DecJan","dsr12DecJan","dsr15DecJan" )

pred.gbm.boruta16 <- predict(object=gbm.ras.boruta16,model=models.boruta$gbm.boruta,na.rm=T)
gc()
plot(pred.gbm.boruta16)
writeRaster(pred.gbm.boruta16,"prediction_boruta_dry2016_gbm.tif",overwrite=TRUE)

rastack.novel.boruta16 <- stack(gbm.ras.boruta16)
novel.raster.boruta16 <- ensemble.novel(x=rastack.novel.boruta16, novel.object=novel.test.boruta)
writeRaster(novel.raster.boruta16,"prediction_boruta_dry2016_novel_gbm.tif",overwrite=TRUE)
##############************************THE END**********************#################

