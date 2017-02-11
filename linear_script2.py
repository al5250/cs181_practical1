from __future__ import division
import pandas as pd
import numpy as np
import math
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.cross_validation import train_test_split
import itertools

df_train = pd.read_csv("train.csv", index_col="smiles", header=0)
print df_train.head()

df_features = df_train.drop(['gap'], axis=1)
y = df_train['gap']

# Get correlations
corrs = []
for column in df_features:
  corrs.append(y.corr(df_features[column]))
good_columns_idx = [not math.isnan(c) for c in corrs]
X_filtered = df_features.iloc[:, good_columns_idx]

def get_xors(df):
  columns = df.columns.values
  print columns
  for c1, c2 in list(itertools.combinations(columns, 2)):
      df[c1 + "xor" + c2] = (df[c1] == 1) ^ (df[c2] == 1)
  return df

X_xors = get_xors(X_filtered)

X_fprints = pd.read_csv("data/fingerprints_train.csv", index_col=0, header=None)
print X_fprints.head()

X = pd.concat([X_xors, X_fprints], axis=1)

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3)

def evaluate(model, X, y):
  accuracy = model.score(X, y)
  print 'Accuracy: {}'.format(accuracy)

print 'Got data'

# Simple Linear Regression Model
LR = LinearRegression()
LR.fit(X_train, y_train)
evaluate(LR, X_train, y_train)
evaluate(LR, X_valid, y_valid)

# Neural Net
nnreg = MLPRegressor(activation='relu', hidden_layer_sizes=[25], alpha=0)
nnreg.fit(X_train, y_train)

evaluate(nnreg, X_train, y_train)
evaluate(nnreg, X_valid, y_valid)

with open("parameters.csv", "w") as myfile:
  myfile.write(str(LR.intercept_))
  for coef in LR.coef_:
    myfile.write(",")
    myfile.write(str(coef))
  myfile.write("\n")

# Ridge Regression Model

evaluate(LR, X_valid, y_valid)

df_fprints_test = pd.read_csv("fingerprints_test.csv", index_col=0, header=None)
print df_fprints_test.head()

yhat_test = LR.predict(df_fprints_test)


# Write to file
with open("predictions.csv", "w") as myfile:
  id = 0
  myfile.write('Id,Prediction\n')
  for pred in yhat_test:
    id += 1
    myfile.write(str(id))
    myfile.write(',')
    myfile.write(str(pred))
    myfile.write('\n')
    if id % 10000 == 0:
      print 'Done with {}'.format(id)
