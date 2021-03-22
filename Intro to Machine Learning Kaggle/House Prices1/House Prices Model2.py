import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = RandomForestRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)

# Load data
prices_file_path = 'C:/Users/dongm/Documents/Kaggle/Intro to Machine Learning Kaggle/house-prices-advanced-regression-techniques/train.csv'
prices_data = pd.read_csv(prices_file_path)

# Features
features = ['OverallCond', '2ndFlrSF', 'OverallQual', 'BsmtUnfSF', '3SsnPorch', 'LotFrontage', 'EnclosedPorch', 'GarageYrBlt', 'BedroomAbvGr',
			'MasVnrArea', 'HalfBath', 'FullBath', 'Fireplaces', 'BsmtFinSF1', 'MiscVal', 'YrSold', 'YearBuilt', 'BsmtFinSF2', 'ScreenPorch',
			'OpenPorchSF', 'BsmtHalfBath', 'MoSold', '1stFlrSF', 'TotRmsAbvGrd', 'GrLivArea', 'YearRemodAdd', 'BsmtFullBath', 'GarageCars',
			'KitchenAbvGr', 'LowQualFinSF', 'GarageArea', 'WoodDeckSF', 'TotalBsmtSF', 'LotArea', 'PoolArea']

# Filter rows with missing values
prices_data = prices_data.dropna(subset = features)

y = prices_data.SalePrice

X = prices_data[features]

# Split the train and test set
train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=0.1, random_state=42)

# Test for efficient tree
min_mae = 10**8
for max_leaf_nodes in [50, 500, 5000, 50000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    if my_mae < min_mae:
    	min_mae = my_mae
    	best_tree_size2 = max_leaf_nodes
    # print("Max leaf nodes: %d  \t\t Mean Absolute Error2:  %d" %(max_leaf_nodes, my_mae))
# print(' ')

# Random Forest Regressor
forest_model_reg = RandomForestRegressor(max_leaf_nodes=best_tree_size2, random_state=1)
forest_model_reg.fit(train_X, train_y)

prices_preds_tr = forest_model_reg.predict(val_X)
error = mean_absolute_error(val_y, prices_preds_tr)
print("Mean Absolute Error:", error)


# Load Data
test_file_path = 'C:/Users/dongm/Documents/Kaggle/Intro to Machine Learning Kaggle/house-prices-advanced-regression-techniques/test.csv'
test_data = pd.read_csv(test_file_path)
sample_file_path = 'C:/Users/dongm/Documents/Kaggle/Intro to Machine Learning Kaggle/sample_submission.csv'
# sample_data = pd.read_csv(sample_file_path)

dic = test_data.mean()

test_data = test_data.fillna(dic)
X_test = test_data[features]
# y_test = sample_data.SalePrice

prices_preds = forest_model_reg.predict(X_test)
# error2 = mean_absolute_error(y_test, prices_preds)
# print("Mean Absolute Error:", error2)

d = {'Id': test_data.Id, 'SalePrice': prices_preds}
result = pd.DataFrame(data=d)
result.to_csv(sample_file_path,index=False)