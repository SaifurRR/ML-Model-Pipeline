import numpy as np
import pandas as pd

from sklearn import svm, datasets
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn import metrics

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

cat_vals = Pipeline([("imputer",SimpleImputer(strategy='most_frequent')), ("ohe",OneHotEncoder(sparse=False, drop='first'))])
num_vals = Pipeline([("imputer",SimpleImputer(strategy='mean')), ("scale",StandardScaler())])

preprocess = ColumnTransformer(
    transformers=[
        ("cat_process", cat_vals, cat_cols),
        ("num_process", num_vals, num_cols)
    ]
)

#Create a pipeline with preprocess and a linear regression model
pipeline = Pipeline([("preprocess",preprocess), 
                     ("regr",LinearRegression())])

#Very simple parameter grid, with and without the intercept
param_grid = {
    "regr__fit_intercept": [True,False]
}

#------------------------------------------------
#1. Grid search using previous pipeline
#grid_search e.g., - > GridSearchCV(classifier, param_grid, cv=5, scoring='accuracy'), cv: cross-validation split
gs=GridSearchCV(pipeline,param_grid,cv=5,scoring='neg_mean_squared_error')

#2. Fit grid using training data and print best score
#fit grid search model:
gs.fit(x_train,y_train)

#3. Calculate best_score
best_score=gs.best_score_ #performance metric of the best estimator during the grid search

#4. Calculate optimal parameters
best_params=gs.best_params_ #hyperparameters that resulted in the best mean cross-validated score during the grid search

print(f'GridSearchCV best score: {best_score}')
print(f'GridSearchCV best params: {best_params}')

