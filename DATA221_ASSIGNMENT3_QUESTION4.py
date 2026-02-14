# Harris Khan
# February 10, 2026
# DATA221, Assignment 3, Question 4

# Import packages
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Load the kidney_disease csv file into a dataframe.
kidney_diseases_data_frame = pd.read_csv("kidney_disease.csv")

#  Remove any rows that contain missing values
kidney_diseases_data_frame = kidney_diseases_data_frame.dropna()

# Creates a feature matrix x which holds all columns except "classifications"
# (axis = 1 means to drop a column, not a row)
feature_matrix_of_kidney_disease = kidney_diseases_data_frame.drop("classification", axis = 1)

# Assigns quantitative values to categorical/qualitative values
feature_matrix_of_kidney_disease = pd.get_dummies(feature_matrix_of_kidney_disease)

# Set classification_label_of_kidney_disease equal the classification column
classification_label_of_kidney_disease = kidney_diseases_data_frame["classification"]
# Set the qualitative values of the classification column into quantitative values
classification_label_of_kidney_disease = classification_label_of_kidney_disease.map({"ckd": 1, "notckd": 0})

# Splits the dataset into training and test subsets
features_train, features_test, labels_train, labels_test = train_test_split(
    feature_matrix_of_kidney_disease,
    classification_label_of_kidney_disease,
    test_size = 0.3, # Model is trained on 30% of the testing data
    random_state = 0 # Create a fixed random state for reproducible results
)

# Set the number of k-nearest neighbors to look for equal to 5
knn_model_of_kidney_disease = KNeighborsClassifier(n_neighbors = 5)

# Train the model using 70% feature data and correct labels
knn_model_of_kidney_disease.fit(features_train, labels_train)

# Predicts the labels of the data values based on the data to be tested
predicted_labels = knn_model_of_kidney_disease.predict(features_test)

# Create a confusion matrix using the real and predicted values of our model.
confusion_matrix_of_classification_model = confusion_matrix(labels_test, predicted_labels)
print("Confusion Matrix:")
print(confusion_matrix_of_classification_model)

# Calculate and print the accuracy of our model using the real and predicted values.
accuracy_of_classification_model = accuracy_score(labels_test, predicted_labels)
print("Accuracy:", accuracy_of_classification_model)

# Calculate and print the precision of our model using the real and predicted values.
precision_of_classification_model = precision_score(labels_test, predicted_labels)
print("Precision:", precision_of_classification_model)

# Calculate and print the recall of our model using the real and predicted values.
recall_of_classification_model = recall_score(labels_test, predicted_labels)
print("Recall:", recall_of_classification_model)

# Calculate and print the f1-score of our model using the real and predicted values.
f1_of_classification_model = f1_score(labels_test, predicted_labels)
print("F1 Score:", f1_of_classification_model)

# Question 1:
# Here are what the four states of the Confusion Matrix would mean in this scenario.
# True Positive: A person with CKD is identified as having CKD.
# True Negative: A person who does not have CKD is identified as not having CKD.
# False Positive: A person who does not have CKD is identified as having CKD.
# False Negative: A person with CKD is identified as not having CKD

# Question 2:
# A high accuracy can be misleading since the accuracy formula is vulnerable to
# being affected by some hidden, confounding variables. Some of the problems that may
# make the accuracy of a model look misleading are imbalanced data, where too much of the
# data is skewed towards one value. An imbalanced dataset could make the model's ability
# to predict values impractical since the model will be more inclined to predict
# towards a certain direction. Another problem that accuracy has is that it can be
# an oversimplification of the model's ability to predict correctly and may leave out
# important information. For more serious tests, such as cancer detection, a false negative
# would be a much more serious matter, however, accuracy does not have any way of telling
# you this.

# Question 3:
# The most important metric for a missing CKD case (False Negative) is the Recall metric. Recall measures
# the 'power' of the model by measuring the total True Positives over the sum of True Positives and False Negatives.
# Through this metric, if a model performs badly and produces too many False Negatives, the recall
# metric can detect this since a higher False Negative value will cause the recall metric to be lower.
# If we were to use accuracy to measure the model's metric for false negatives, it would not
# work quite as well. Since accuracy is more of a generalization of the model performance due to the
# greater number of variables used within its formula, it does not measure a specific aspect of the
# model and may miss crucial shortcomings of the model.