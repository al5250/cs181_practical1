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

df_fprints = pd.read_csv("fingerprints_train.csv", index_col=0, header=None)
print df_fprints.head()

y = df_train['gap']

def get_xors(df):
  columns = df.columns.values
  print columns
  for c1, c2 in list(itertools.combinations(columns, 2)):
      df[c1 + "and" + c2] = (df[c1] == 1) & (df[c2] == 1)
      df[c1 + "xor" + c2] = (df[c1] == 1) ^ (df[c2] == 1)
  return df

X_train, X_valid, y_train, y_valid = train_test_split(df_fprints, y, test_size=0.3)

def evaluate(model, X, y):
  accuracy = model.score(X, y)
  print 'Accuracy: {}'.format(accuracy)

# Simple Linear Regression Model
LR = LinearRegression()
LR.fit(X_train, y_train)
evaluate(LR, X_train, y_train)

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
