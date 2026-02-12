# Harris Khan
# February 10, 2026
# DATA221, Assignment 3, Question 3

# Import packages
import pandas as pd
from sklearn.model_selection import train_test_split

# Load the kidney_disease csv file into a dataframe
kidney_diseases_data_frame = pd.read_csv("kidney_disease.csv")

# Creates a feature matrix x which holds all columns except "classifications"
# (axis = 1 means to drop a column, not a row)
x = kidney_diseases_data_frame.drop("classification", axis = 1)

y = kidney_diseases_data_frame["classification"]

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.3,      # 30% testing
    random_state=0     # fixed random state for reproducibility
)