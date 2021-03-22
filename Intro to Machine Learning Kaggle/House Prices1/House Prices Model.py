import random
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

def get_mae1(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)

def get_mae2(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = RandomForestRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)

# Load data
prices_file_path = 'C:/Users/dongm/OneDrive/Academia/Intro to Machine Learning Kaggle/house-prices-advanced-regression-techniques/train.csv'
prices_data = pd.read_csv(prices_file_path)
# test_file_path = 'C:/Users/dongm/OneDrive/Academia/Intro to Machine Learning Kaggle/house-prices-advanced-regression-techniques/test.csv'
# test_data = pd.read_csv(test_file_path)

# Features
# Features with numbers
features1 = ['LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1',
       		 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 
       		 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces',
       		 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch',
       		 'PoolArea', 'MiscVal', 'MoSold', 'YrSold']
# length of features1 = 35

# Features with words
features2 = ['MSSubClass', 'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 
			 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 
			 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 
			 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 
			 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 
			 'MiscFeature', 'SaleType', 'SaleCondition']

min_error1 = 10**8
min_error2 = 10**8
for i in range(10):
	# Random choose features
	# k = random.randint(1,35)
	# print(k)
	k = 35
	features = random.sample(features1,k)
	# print(features)
	# print(' ')

	# Filter rows with missing values
	prices_data = prices_data.dropna(subset = features)
	# test_data = test_data.dropna(axis=0,how='all')

	y = prices_data.SalePrice

	X = prices_data[features]

	# Split the train and test set
	train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=0.1, random_state=42)

	# Test for efficient tree
	min_mae1 = 10**8
	for max_leaf_nodes in [50, 430, 450, 500, 5000]:
	    my_mae1 = get_mae1(max_leaf_nodes, train_X, val_X, train_y, val_y)
	    if my_mae1 < min_mae1:
	    	min_mae1 = my_mae1
	    	best_tree_size1 = max_leaf_nodes
	    # print("Max leaf nodes: %d  \t\t Mean Absolute Error1:  %d" %(max_leaf_nodes, my_mae1))
	# print(' ')

	min_mae2 = 10**8
	for max_leaf_nodes in [50, 500, 5000, 50000]:
	    my_mae2 = get_mae2(max_leaf_nodes, train_X, val_X, train_y, val_y)
	    if my_mae2 < min_mae2:
	    	min_mae2 = my_mae2
	    	best_tree_size2 = max_leaf_nodes
	    # print("Max leaf nodes: %d  \t\t Mean Absolute Error2:  %d" %(max_leaf_nodes, my_mae2))
	# print(' ')


	# Decision Tree Regressor
	prices_model_reg = DecisionTreeRegressor(max_leaf_nodes=best_tree_size1, random_state=1)
	prices_model_reg.fit(train_X,train_y)

	prices_preds1_reg = prices_model_reg.predict(val_X)
	error1 = mean_absolute_error(val_y, prices_preds1_reg)
	print("Mean Absolute Error1:", mean_absolute_error(val_y, prices_preds1_reg))


	# Random Forest Regressor
	forest_model_reg = RandomForestRegressor(max_leaf_nodes=best_tree_size2, random_state=1)
	forest_model_reg.fit(train_X, train_y)

	prices_preds2_reg = forest_model_reg.predict(val_X)
	error2 = mean_absolute_error(val_y, prices_preds2_reg)
	print("Mean Absolute Error2:", mean_absolute_error(val_y, prices_preds2_reg))
	print(' ')

	if error1 < min_error1:
		min_error1 = error1
		c1 = k
		f1 = features

	if error2 < min_error2:
		min_error2 = error2
		c2 = k
		f2 = features

print("Features:", f1)
print("num: %d  \t\t Mean Absolute Error1:  %d \n" %(c1, min_error1))
print("Features:", f2)
print("num: %d  \t\t Mean Absolute Error2:  %d" %(c2, min_error2))