import pandas as pd

# save filepath to variable for eeasier access
melbourne_file_path = 'C:/Users/dongm/OneDrive/Academia/Intro to Machine Learning Kaggle/melb_data.csv'

# read the data and store data in DataFram titled melbourne_data
melbourne_data = pd.read_csv(melbourne_file_path)

# print a summary of the data in Melbourne data
pd.set_option('max_columns', None)
print(melbourne_data.describe())