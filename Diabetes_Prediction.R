# Loading required libraries
library(caret)
library(ggplot2)
library(ROCR)
library(dplyr)
library(e1071)
library(caTools)

# Reading the dataset
data <- read.csv("https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv", header = FALSE)

# Renaming columns for better clarity
colnames(data) <- c("Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome")

# Displaying the first few rows and structure of the dataset
head(data)
str(data)
summary(data)

# Checking for missing values
colSums(is.na(data))

# Replacing zeros with NA for certain columns where zero is not realistic
data$Glucose[data$Glucose == 0] <- NA
data$BloodPressure[data$BloodPressure == 0] <- NA
data$SkinThickness[data$SkinThickness == 0] <- NA
data$Insulin[data$Insulin == 0] <- NA
data$BMI[data$BMI == 0] <- NA

# Imputing missing values with median of each column
data <- data %>%
  mutate(Glucose = ifelse(is.na(Glucose), median(Glucose, na.rm = TRUE), Glucose),
         BloodPressure = ifelse(is.na(BloodPressure), median(BloodPressure, na.rm = TRUE), BloodPressure),
         SkinThickness = ifelse(is.na(SkinThickness), median(SkinThickness, na.rm = TRUE), SkinThickness),
         Insulin = ifelse(is.na(Insulin), median(Insulin, na.rm = TRUE), Insulin),
         BMI = ifelse(is.na(BMI), median(BMI, na.rm = TRUE), BMI))

# Scaling the data for better model performance
scaled_data <- as.data.frame(scale(data[,-9]))
scaled_data$Outcome <- data$Outcome

# Performing exploratory data analysis (EDA)
# Ploting histograms for each feature
feature_names <- names(scaled_data)[-which(names(scaled_data) %in% c("Outcome"))]
par(mfrow = c(3, 3))  # Arrange plots in a 3x3 grid
for (feature in feature_names) {
  hist(scaled_data[[feature]], main = feature, xlab = feature, col = "lightblue", border = "black")
}

# Ploting correlation matrix
plot.new()
dev.off()  # Ensure a clean plotting environment
corr_matrix <- cor(scaled_data[, -9])
corrplot(corr_matrix, method = "circle")

# Ploting boxplots to visualize feature distribution by outcome
par(mfrow = c(3, 3))
for (feature in feature_names) {
  boxplot(scaled_data[[feature]] ~ scaled_data$Outcome, main = feature, xlab = "Outcome", col = c("lightblue", "lightgreen"))
}

# Spliting data into training and testing sets
set.seed(123)
split <- sample.split(scaled_data$Outcome, SplitRatio = 0.7)
train_data <- subset(scaled_data, split == TRUE)
test_data <- subset(scaled_data, split == FALSE)

# Building a logistic regression model
logistic <- glm(Outcome ~ ., data = train_data, family = "binomial")
summary(logistic)

# Making predictions on test data
predicted_prob <- predict(logistic, newdata = test_data, type = "response")
predicted_clas <- ifelse(predicted_prob > 0.2, 1, 0)  # Threshold set to 0.2

# Evaluating model performance using confusion matrix
plot.new()
dev.off()
confusionMatrix(table(actual_data = test_data$Outcome, predicted_data = predicted_clas))

# Calculating and ploting ROC curve
pred <- prediction(predicted_prob, test_data$Outcome)
perf <- performance(pred, "tpr", "fpr")
plot(perf, colorize = TRUE)
auc <- performance(pred, measure = "auc")
auc <- auc@y.values[[1]]
print(paste("AUC:", auc))
