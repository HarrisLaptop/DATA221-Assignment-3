# Harris Khan
# February 10, 2026
# DATA221, Assignment 3, Question 3

# Import packages
import pandas as pd
from sklearn.model_selection import train_test_split

# Load the kidney_disease csv file into a dataframe
kidney_diseases_data_frame = pd.read_csv("kidney_disease.csv")

#  Remove any rows that contain missing values
kidney_diseases_data_frame = kidney_diseases_data_frame.dropna()

# Creates a feature matrix x which holds all columns except "classifications"
# (axis = 1 means to drop a column, not a row)
feature_matrix_of_kidney_disease = kidney_diseases_data_frame.drop("classification", axis = 1)

# Assigns quantitative values to categorical/qualitative values
feature_matrix_of_kidney_disease = pd.get_dummies(feature_matrix_of_kidney_disease)


classification_label_of_kidney_disease = kidney_diseases_data_frame["classification"]
# Set the qualitative values of the classification column into quantitative values
classification_label_of_kidney_disease = classification_label_of_kidney_disease.map({"ckd": 1, "notckd": 0})

features_train, features_test, labels_train, labels_test = train_test_split(
    feature_matrix_of_kidney_disease,
    classification_label_of_kidney_disease,
    test_size=0.3,     # Model is trained on 30% of the testing data
    random_state=0     # Create a fixed random state for reproducible results
)

# Question 1:
# We should not train and test a model on the same data since the model won't actually learn anything
# through this approach. This method would only test how well it can memorize the data set. This causes
# an overfitting problem by memorizing the data instead of learning patterns. An example to illustrate
# why this could be a problem is like if you were to study a test by only memorizing the answers on a
# test. You might get 100% if the test contains those exact same answer, but if there are any new questions,
# you won't know how to answer them.

# Question 2:
# The purpose of the testing set is to measure how well the model works under new, unseen data. The goal is
# to increase the performance of the model after training it on some of the known data. For example, if
# the training accuracy is high, but the testing accuracy is low, the model is overfitted and needs more
# testing data to increase its accuracy. To illustrate this point, imagine the process of studying for an
# exam as the training part of the model, and the process of writing the exam as the testing part of the
# model. You aren't graded by how well you study, you're graded by how well you do on the exam. 