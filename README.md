This Project is was created to compare a Linear Regression model from WEKA with XGBoost.
XGBoost regression does poorly on time - series based data, thus we approached the solution by creating a
n-step ahead algorithm to predict every y label.

Components invovled are:
	- Normalization
	- PCA
	- N Step ahead data processing
	- N Step ahead model creation
	- XGBoost model initialization based on docs