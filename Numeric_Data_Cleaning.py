import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# perform imputation (replace missing values) -> standard scaler
## Loading the dataset
columns = ["sex","length","diam","height","whole","shucked","viscera","shell","age"]
df = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data",names=columns)
## Defining target and predictor variables
y = df.age #target
X = df.drop(columns=['age'])

## Numerical columns:
num_cols = X.select_dtypes(include=np.number).columns
## Categorical columns
cat_cols = X.select_dtypes(include=['object']).columns

## Create some missing values randomly in dataset:
## test how well different data handling strategies e.g. imputation methods perform in presence of missing values.
for i in range(1000):
    X.loc[np.random.choice(X.index),np.random.choice(X.columns)] = np.nan

## Perform train-test split
x_train, x_test, y_train, y_test = train_test_split(X,y, random_state=0, test_size=0.25)

#####-------Imputation and Scaling: Code base to transform -----------------#####
## Numerical training data
x_train_num = x_train[num_cols]
# Filling in missing values with mean on numeric features only
x_train_fill_missing = x_train_num.fillna(x_train_num.mean())
## Fitting standard scaler on x_train_fill_missing
scale = StandardScaler().fit(x_train_fill_missing)
## Scaling data after filling in missing values
x_train_fill_missing_scale = scale.transform(x_train_fill_missing)
## Same steps as above, but on the test set:
x_test_fill_missing = x_test[num_cols].fillna(x_train_num.mean())
x_test_fill_missing_scale = scale.transform(x_test_fill_missing)
#####-------Imputation and Scaling: Code base to transform -----------------#####

#1. Rewrite using Pipelines!
pipeline=Pipeline([("imputer", SimpleImputer(strategy='mean')),("scale", StandardScaler())])

##2. Fit pipeline on the test and compare results
pipeline.fit(x_train[num_cols])
x_transform=pipeline.transform(x_test[num_cols])
#3.  Verify pipeline transform test set is the same by using np.array_equal()
array_diff=np.array_equal(x_transform,x_test_fill_missing_scale)
print(f'pipeline_arr == np_arr: {array_diff}')

#4. Change imputer strategy to median 
pipeline_median=Pipeline([('imputer', SimpleImputer(strategy='median')),('scale',StandardScaler())])

# 5 Compare results between the two pipelines
pipeline_median.fit(x_train[num_cols])
x_transform_median=pipeline_median.transform(x_test[num_cols])

new_array_diff=abs(x_transform-x_transform_median).sum()
print(f'pipeline_arr_med - np_arr_med = {new_array_diff}')
