import numpy as np
import pandas as pd

from sklearn import svm, datasets
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score

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
#1. Create a pipeline with `preprocess` and a linear regression model, `regr`
pipeline=Pipeline([("preprocess", preprocess),("regr ", LinearRegression())]) # cat: simple_impute, ohe, regr ; num: simple_impute, standardscale, regr

#2. Fit the pipeline on the training data and predict on the test data
pipeline.fit(x_train, y_train)
#prediction on test data
y_pred=pipeline.predict(x_test)

#3. Calculate pipeline score and compare to estimator score
#Pipeline score
pipeline_score=pipeline.score(x_test,y_test) #default is R^2 score
print(f'pipeline score: {pipeline_score}')

#r-squared score
r2_score = r2_score(y_test, y_pred)
print(f'sklearn metric: {r2_score}')
