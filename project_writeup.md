#Determining Fitness Exercise Correctness Using a Machine Learning Based Predictive Model

## Introduction

This project is aimed at using R programming language, Machine Learning algorithms and some provided dataset which describes how clients performs excercises
. The dataset is divided into 2 sets called training and test set. The training dataset describes the activities during the exercise and determines the correctness of the overall process
by assigning a grade "A", "B","C","D" or "E" as shown in the "classe" column. while the test set contains activities of new client but
is aimed at grading them based on the experience of a machine learning model which has been trained using a the training set.

To achive this, there is need for the following:

* Data aquisition
* Data Exploration
* Building of Model
* Testing of Model

## Data aquisition
The training and testing dataset has been provided by [1] which can be found in the following links.

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

This can be read into R programming laguage as follows:
```
training <- read.csv('pml-training.csv')

testing <- read.csv('pml-testing.csv')
```
## Data Exploration
Data exploration helps in cleaning the data, removing irrelevant features (predictors) or features which are highly correlated to another feature. It also helps in identifying missing predictors. This helps to prevent overfitting the data. 

In this case, the following features below are not relevant to the correctness of the clients excersice but may be useful for administrative purpose hence they are removed.
```
X, user_name, raw_timestamp_part1, raw_timestamp_part2, cvtd_timestamp, new_window and num_window.
```
Also, features whose values are not available (e.g NA) are removed as they provide no useful information. hence, the selected relevant features are:

```R
selected_features <- c('roll_belt', 'pitch_belt', 'yaw_belt', 'total_accel_belt',
                  'gyros_belt_x', 'gyros_belt_y', 'gyros_belt_z',
                  'accel_belt_x', 'accel_belt_y', 'accel_belt_z',
                  'magnet_belt_x', 'magnet_belt_y', 'magnet_belt_z',
                  'roll_arm', 'pitch_arm', 'yaw_arm', 'total_accel_arm',
                  'gyros_arm_x', 'gyros_arm_y', 'gyros_arm_z',
                  'accel_arm_x', 'accel_arm_y', 'accel_arm_z',
                  'magnet_arm_x', 'magnet_arm_y', 'magnet_arm_z',
                  'roll_dumbbell', 'pitch_dumbbell', 'yaw_dumbbell', 'total_accel_dumbbell',
                  'gyros_dumbbell_x', 'gyros_dumbbell_y', 'gyros_dumbbell_z',
                  'accel_dumbbell_x', 'accel_dumbbell_y', 'accel_dumbbell_z',
                  'magnet_dumbbell_x', 'magnet_dumbbell_y', 'magnet_dumbbell_z',
                  'roll_forearm', 'pitch_forearm', 'yaw_forearm', 'total_accel_forearm',
                  'gyros_forearm_x', 'gyros_forearm_y', 'gyros_forearm_z',
                  'accel_forearm_x', 'accel_forearm_y', 'accel_forearm_z',
                  'magnet_forearm_x', 'magnet_forearm_y', 'magnet_forearm_z'
                  )
```

The refined training and testing set is now given as follows:

```
testing <- testing[, selected_features]
selected_features_trainig <- c(selected_features, 'classe')
training <- training[, selected_features_training]
```

### Identifying highly correlated features
A pairs-wise correlation is done among the features using Pearson's correlation test to identify highly correlated features (both to the negative and positive sides) as shown below:
```
corr <- cor(training[, names(training) != 'classe'])
corr[(corr < -0.8 | corr > 0.8) & corr != 1]
```
The test shows there are about 19 correlated fetures which are above an absolute value of 0.8. However, further increase in the threshold shows limits the number of highly correlated pairs to 3 as shown below.
```
which(corr > 0.98 & corr != 1)
corr[which(corr > 0.98 & corr != 1)]
which(corr < -0.98)
pred.corr[which(corr < -0.98)]
```
Hence, discarding one of the features among this pairs is needed. In this case the "roll_belt" is chosen to be discarded.
```
training <- subset(training, select=-c(roll_belt))
testing <- subset(testing, select=-c(roll_belt))
```
## Building and Testing the Model
Random forest algorithm was chosen for this task because of the following reasons:
* Robustness to outliers in input space
* Computational Scalability
* Ability to handle missing values
* Natural Handling of data of mixed types

The classification and testing is a shown below:

```R
library(randomForest)
library(caret)
library(grDevices)

set.seed(2222)



model <- randomForest(classe ~ ., data = training) # training the model
model # checking the accuracy on the training set, out of bag error (oob) and confusion matrix 

predict(model, testing)  # performing prediction on the test set
```
# Result
This model performs very well (100% accuracy) and an out of bag error (oob) of 0.32%.

# References
1. Ugulino, W.; Cardador, D.; Vega, K.; Velloso, E.; Milidiu, R.; Fuks, H. Wearable Computing: Accelerometers' Data Classification of Body Postures and Movements. Proceedings of 21st Brazilian Symposium on Artificial Intelligence. Advances in Artificial Intelligence - SBIA 2012. In: Lecture Notes in Computer Science., pp. 52-61. Curitiba, PR: Springer Berlin / Heidelberg, 2012. ISBN 978-3-642-34458-9. DOI: 10.1007/978-3-642-34459-6_6. 
