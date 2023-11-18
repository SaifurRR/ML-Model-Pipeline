import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import KFold, train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin

columns = ["sex","length","diam","height","whole","shucked","viscera","shell","age"]
df = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data",names=columns)
y = df.age
X=df.drop(columns=['age'])
num_cols = X.select_dtypes(include=np.number).columns
cat_cols = X.select_dtypes(include=['object']).columns

for i in range(1000):
    X.loc[np.random.choice(X.index),np.random.choice(X.columns)] = np.nan
x_train, x_test, y_train, y_test = train_test_split(X,y, random_state=0, test_size=0.25)
x_train_num = x_train[num_cols]

#fill missing values with mean on numeric features only
x_train_fill_missing = x_train_num.fillna(x_train_num.mean())

#fit standard scaler on x_train_fill_missing
scale = StandardScaler().fit(x_train_fill_missing)

#scale data after filling in missing values
x_train_fill_missing_scale = scale.transform(x_train_fill_missing)
x_test_fill_missing = x_test[num_cols].fillna(x_train_num.mean())
x_test_fill_missing_scale = scale.transform(x_test_fill_missing)

class MyImputer(BaseEstimator, TransformerMixin): 
    def __init__(self):
        return None
    
    def fit(self, X, y = None):
        self.means = np.mean(X, axis=0)    # calculate the mean of each column
        return self
    
    def transform(self, X, y = None):
        #transform method fills in missing values with means using pandas
        return X.fillna(self.means)

#1. Create a new pipeline using the custom class MyImputer as the first step and standard scaler on the second
new_pipeline=Pipeline([("my_imputer", MyImputer()),("scale", StandardScaler())])

#2. Fit new pipeline on the training data with num_cols only and verify that the results of the transform are the same on test set
#fit train set
new_pipeline.fit(x_train[num_cols])

#transform test set
x_transform=new_pipeline.transform(x_test[num_cols])

#compare the arrays (train & test)
check_arrays=np.array_equal(x_transform,x_test_fill_missing_scale)
print(f'check custom imputer & fillna: {check_arrays}')
