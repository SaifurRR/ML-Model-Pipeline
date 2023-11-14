#import libraries
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

columns = ["sex","length","diam","height","whole","shucked","viscera","shell","age"]
df = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data",names=columns)

y = df.age
X=df.drop(columns=['age'])
num_cols = X.select_dtypes(include=np.number).columns
cat_cols = X.select_dtypes(include=['object']).columns

#create some missing values
for i in range(1000):
    X.loc[np.random.choice(X.index),np.random.choice(X.columns)] = np.nan

x_train, x_test, y_train, y_test = train_test_split(X,y, random_state=0, test_size=0.25)

#1. Create numerical and categorical pipelines called `num_vals` and `cat_vals`

#numerical pipelines:
num_vals=Pipeline([("imputer", SimpleImputer(strategy='mean')),("scale", StandardScaler())])
#categorical pipelines:
cat_vals=Pipeline([("imputer", SimpleImputer(strategy='most_frequent')),("ohe", OneHotEncoder(drop = 'first',sparse = False))])

#2. Create the column transformer with the categorical and numerical processes

# ColumnTransformer(transformers=[(name1, pipeline1, col1),(name2, pipeline2, col2)])
preprocess =ColumnTransformer(transformers=[('num_preprocess',num_vals,num_cols),('cat_preprocess',cat_vals,cat_cols)])

#3. Fit the preprocess transformer to training data

x_train_=preprocess.fit(x_train)
x_transform=preprocess.transform(x_test)
print(f'preprocess transformer to training data: \n {x_train_}\n')

#4. Store transformed test data

print(f'store transformed test data: \n {x_transform}\n')
