import numpy as np
import pandas as pd
import seaborn as sns
sns.set_palette('husl')
import matplotlib.pyplot as plt
# %matplotlib inline
import torch

from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

datapath='./input'
data = pd.read_csv(datapath+'/train.csv',error_bad_lines=False)
test= pd.read_csv(datapath+'/test.csv')

test_X=test.drop(['id'],axis=1)
test_Y=test['id']

data = data.apply(pd.to_numeric, errors='coerce')
data.info()

data=data.dropna()

X=data.drop(['intervention'],axis=1)
y=data['intervention']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

#adaboost=AdaBoostClassifier(n_estimators=500,learning_rate=0.5)
#adaboost.fit(X_train,y_train)
#Y_pred = adaboost.predict(X_test)

gbc=GradientBoostingClassifier(learning_rate=0.01)
gbc.fit(X_train,y_train)
Y_pred = gbc.predict(X_test)
print("gbc accuracy_score: {}".format( accuracy_score(y_test, Y_pred)))

#print("adaboost accuracy_score: {}".format( accuracy_score(y_test, Y_pred)))

#ra_pred=adaboost.predict(test_X)

ra_pred=gbc.predict(test_X)
ra_pred=ra_pred.astype(int)

submission = pd.DataFrame({
        "id": test["id"],
        "intervention": ra_pred
        })
submission.to_csv('gbc_sub.csv',index=False)