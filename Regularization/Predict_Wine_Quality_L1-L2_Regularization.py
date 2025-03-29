#import libraries
import numpy as np
import pandas as pd
import codecademylib3
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score

# 1. Data Pre-processing
df = pd.read_csv('wine_quality.csv')
y = df['quality']
features = df.drop(columns = ['quality'])

# 2. Data transformation
scaler = StandardScaler()
df_scaled = scaler.fit(features)
X = df_scaled.transform(features)

# 3. Train-test split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 99)

# 4. Fit a logistic regression classifier without regularization
clf_no_reg = LogisticRegression(penalty = 'none')
clf_no_reg.fit(x_train, y_train)

# 5. Plot the coefficients
predictors = features.columns
coefficients = clf_no_reg.coef_.ravel() #flatten 2D -> 1D arr
coef = pd.Series(coefficients,predictors).sort_values()
coef.plot(kind='bar', title = 'Coefficients (with no regularization)')
plt.tight_layout() #prevent label cutoffs
plt.show()  
plt.clf() #prevent overlaps in subsequent plots

# 6. Training and Test performance (Classifier), f1-score 
y_pred_train = clf_no_reg.predict(x_train)
y_pred_test = clf_no_reg.predict(x_test)
f1_train = f1_score(y_train, y_pred_train)
f1_test = f1_score(y_test, y_pred_test)
print('Training Score: ' , f1_train)
print('Test Score: ' , f1_test)

# 7. Default Implementation (L2-regularized!)
clf_L2_reg = LogisticRegression()
clf_L2_reg.fit(x_train, y_train)

# 8. L2, Ridge f2-Scores
f1_L2_train = f1_score(y_train, clf_L2_reg.predict(x_train))
f1_L2_test = f1_score(y_test, clf_L2_reg.predict(x_test))
print('Training Score (L2):', f1_L2_train)
print('Test Score (L2):', f1_L2_test) #test val, remains the same

# 9. Coarse-grained hyperparameter tuning for more regularization
training_array = []
test_array = []
C_array = [0.0001, 0.001, 0.01, 0.1, 1]
for c in C_array:
  clf_L2_reg = LogisticRegression(C=c)
  clf_L2_reg.fit(x_train, y_train)
  training_array.append(f1_score(y_train, clf_L2_reg.predict(x_train)))
  test_array.append(f1_score(y_test, clf_L2_reg.predict(x_test)))

# 10. Plot training and test scores as a function of C
plt.plot(C_array,training_array, label ='training')
plt.plot(C_array,test_array, label ='test')
plt.xscale('log')
plt.xlabel('C - coarse')
plt.ylabel('f1 - score')
plt.legend()
plt.show()
plt.clf()

# 11. Hyperparameter Tuning for L2 Regularization - GridSearchCV
tuning_C = {'C' : np.logspace(-4, -2, 100)}

# 12. Implementing GridSearchCV with l2 penalty
grid_search = GridSearchCV(LogisticRegression(), param_grid = tuning_C, scoring = 'f1', cv = 5)
grid_search.fit(x_train, y_train)

# 13. Optimal C value and the score corresponding to it
best_C = grid_search.best_params_['C']
best_score = grid_search.best_score_
print(grid_search.best_params_, grid_search.best_score_)

# 14. Validating the "best classifier" - use 'test data' as validation
clf_best_ridge = LogisticRegression(C = best_C)
clf_best_ridge.fit(x_train, y_train)
f1_score_bestparam = f1_score(y_test, clf_best_ridge.predict(x_test))

# 15. Implement L1 hyperparameter tuning with LogisticRegressionCV
clf_l1 = LogisticRegressionCV(Cs = np.logspace(-2, 2, 100), penalty = 'l1', cv = 5, solver = 'liblinear', scoring = 'f1')
clf_l1.fit(x_train, y_train)

# 16. Optimal C value and corresponding coefficients
print('Best C value:', clf_l1.C_ )
print('Best-fit coefficients:', clf_l1.coef_ )

# 17. Plotting the tuned L1 coefficients
coefficients = clf_l1.coef_.ravel()
coef = pd.Series(coefficients,predictors).sort_values()
plt.figure(figsize = (12,8))
coef.plot(kind='bar', title = 'Coefficients for tuned L1')
plt.tight_layout()
plt.show()
plt.clf()
