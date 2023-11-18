import numpy as np
import pandas as pd

from sklearn import svm, datasets
from sklearn.linear_model import LinearRegression, Lasso, Ridge
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
        ("cat_preprocess", cat_vals, cat_cols),
        ("num_preprocess", num_vals, num_cols)
    ]
)
#Create a pipeline with preprocess and a linear regression model
pipeline = Pipeline([("preprocess",preprocess), 
                     ("regr",LinearRegression())])

#1. Update the `search_space` array from the narrative to add a Lasso Regression model as the third dictionary.
search_space = [{'regr': [LinearRegression()], 'regr__fit_intercept': [True,False]},
                {'regr':[Ridge()],
                     'regr__alpha': [0.01,0.1,1,10,100]},
                {'regr':[Lasso()],
                     'regr__alpha': [0.01,0.1,1,10,100]}]

#2.  Initialize a grid search on `search_space`
gs = GridSearchCV(pipeline, search_space, scoring='neg_mean_squared_error', cv=5)

#3. Find the best pipeline, regression model and its hyperparameters

# i. Fit to training data
gs.fit(x_train, y_train)

# ii. Find the best pipeline
best_pipeline = gs.best_estimator_

# iii. Find the best regression model
best_regression_model = best_pipeline.named_steps['regr']
print('The best regression model is:')
print(best_regression_model)

# iv. Find the hyperparameters of the best regression model
best_model_hyperparameters = best_regression_model.get_params()
print('The hyperparameters of the regression model are:')
print(best_model_hyperparameters)

#4. Access the hyperparameters of the categorical preprocessing step
cat_preprocess_hyperparameters = best_pipeline.named_steps['preprocess'].named_transformers_['cat_preprocess'].named_steps['imputer'].get_params()
print('The hyperparameters of the imputer are:')
print(cat_preprocess_hyperparameters)  

