import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

columns = ["sex","length","diam","height","whole","shucked","viscera","shell","age"]
df = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data",names=columns)

y = df.age
X=df.drop(columns=['age'])
#numerical cols
num_cols = X.select_dtypes(include=np.number).columns
#categorical cols
cat_cols = X.select_dtypes(include=['object']).columns
#create some missing values
for i in range(1000):
    X.loc[np.random.choice(X.index),np.random.choice(X.columns)] = np.nan

x_train, x_test, y_train, y_test = train_test_split(X,y, random_state=0, test_size=0.25)
x_train_cat = x_train[cat_cols]

#Fill missing values with mode on categorical features only: fill with 'M'
x_train_fill_missing = x_train_cat.fillna(x_train_cat.mode().values[0][0]) #2d arr -> [row][col]

#Apply one hot encoding on x_train_fill_missing
ohe = OneHotEncoder(sparse=False, drop='first').fit(x_train_fill_missing) #every element is displayed -> 0 1

#Transform data after filling in missing values
x_train_fill_missing_ohe = ohe.transform(x_train_fill_missing)

#now we want to do the same thing on the test set! 
x_test_fill_missing = x_test[cat_cols].fillna(x_train_cat.mode().values[0][0])
x_test_fill_missing_ohe = ohe.transform(x_test_fill_missing)

#1. Rewrite using Pipelines!
pipeline=Pipeline([("imputer", SimpleImputer(strategy='most_frequent')),("one_hot_encode", OneHotEncoder(drop = 'first',sparse = False))])

#2. Fit the pipeline and transform the test data (categorical columns only!)
pipeline.fit(x_train[cat_cols])

x_transform=pipeline.transform(x_test[cat_cols])

#3. Check if the two arrays are the same using np.array_equal()
check_arrays= np.array_equal(x_transform,x_test_fill_missing_ohe)

print(f'pipeline_arr == np_arr : {check_arrays}')

