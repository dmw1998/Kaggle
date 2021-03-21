import pandas as pd

melbourne_file_path = 'C:/Users/dongm/OneDrive/Academia/Intro to Machine Learning Kaggle/melb_data.csv'
melbourne_data = pd.read_csv(melbourne_file_path)
print(melbourne_data.columns)

# dropna drops missing values (think of na as "not available")
melbourne_data = melbourne_data.dropna(axis=0)

# select a column
y = melbourne_data.Price

# select multipe features
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X = melbourne_data[melbourne_features]
print(X.describe())

# return the first n (= 5) rows
print(X.head())

## Building a Model
from sklearn.tree import DecisionTreeRegressor

# Define model. Specify a number from random_state to ensure same results each run
melbourne_model = DecisionTreeRegressor(random_state=1)

# Fit model
print(melbourne_model.fit(X,y))

print("Making predictions for the following 5 houses:")
print(X.head())
print("The predictions are")
print(melbourne_model.predict(X.head()))