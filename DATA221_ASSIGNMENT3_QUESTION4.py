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