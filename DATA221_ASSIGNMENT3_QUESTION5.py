# Harris Khan
# February 10, 2026
# DATA221, Assignment 3, Question 5

# Import required packages
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

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

# Create a list of k-values and an empty accuracy list for the table to be created below
k_values = [1,3,5,7,9]
accuracy_list = []

# For every k-value
for num_neighbors in k_values:
    # Set the number of k-nearest neighbors to look for equal to the current k-value
    knn_model_of_kidney_disease = KNeighborsClassifier(n_neighbors = num_neighbors)

    # Train the model using 70% feature data and correct labels
    knn_model_of_kidney_disease.fit(features_train, labels_train)

    # Predicts the labels of the data values based on the data to be tested
    predicted_labels = knn_model_of_kidney_disease.predict(features_test)

    # Compute the accuracy of the model and append its value to the list of each k-value's accuracy
    accuracy = accuracy_score(labels_test, predicted_labels)
    accuracy_list.append(accuracy)

# Create a table containing each k-value's corresponding accuracy
accuracy_and_k_value_table = pd.DataFrame({
    "k value": k_values,
    "Accuracy": accuracy_list
})

# Print the table
print(accuracy_and_k_value_table)

# Locate the row with the highest accuracy and store it in this variable
best_k_value = accuracy_and_k_value_table.loc[accuracy_and_k_value_table["Accuracy"].idxmax()]
print("\nThe value of k that gives the highest test accuracy is", int(best_k_value["k value"]))

# Question 1:
# Changing the k-value affects how much influence the model places on each nearest neighbor.
# If we use a smaller k-value, then each neighbor that the model uses to train itself has a
# greater influence in training the model. The model also becomes much more sensitive to
# messy, 'noisy' data since it may not be able to find general patterns (it may get too stuck
# in the details). A larger k-value places less emphasis on each nearest neighbor and takes a
# broad approach to looking at the data.

# Question 2:
# Small values of k may cause overfitting since the model may base its predictions off of
# limited information. This may be a problem since the model might not be able to detect general
# trends and patterns. Smaller k-values may also be more sensitive to noise since the inability
# to find general patterns means that more emphasis is placed on the noisy data.

# Question 3:
# Large values of k may cause underfitting since the model may base its predictions off of
# too much information. This may be a problem since the model may ignore important patterns
# by not picking up important details since less emphasis is placed on each nearest neighbor.
# Although a higher k-value may help to reduce the impact that noisy data may have, it may also
# oversimplify the model and struggle to notice any patterns if the k-value is too high.
