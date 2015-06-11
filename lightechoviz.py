import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import numpy as np
import sys
sys.path.append("code/")

from sklearn.preprocessing import StandardScaler
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import KFold
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn import linear_model, decomposition, datasets

from echostate import EchoStateNetwork, EchoStateEnsemble
import datagen

n1 = np.loadtxt("data/narma1.dat")
n2 = np.loadtxt("data/narma2.dat")
n3 = np.loadtxt("data/narma3.dat")

n1_states = ["n1" for i in xrange(n1.shape[0])]
n2_states = ["n2" for i in xrange(n2.shape[0])]
n3_states = ["n3" for i in xrange(n3.shape[0])]

Y = np.vstack((n1, n2, n3))
states = np.hstack((n1_states, n2_states, n3_states))

## scaling all the data
Y_scaled = StandardScaler().fit_transform(Y.T).T

## make the pipeline
#pipeline = Pipeline([("ess", EchoStateEnsemble(n_washout=100,scaling=1.0,lamb=1.e-4, topology="scr")),
#                     ("rfc", RandomForestClassifier(n_estimators=300))])


## set the parameters to search over
#params = {"ess__N":[10,20,50,100,200,500],
#          "ess__a":[0.01,0.1,0.2,0.5,0.75,1.0],
#          "ess__r":[0.01, 0.1, 0.2, 0.5, 0.75, 1.0],
#          "rfc__max_depth":[5,10,15,20,50,100,200,500]}

#logistic = linear_model.LogisticRegression()
esn = EchoStateEnsemble(n_washout=100, scaling=1.0, lamb=1.e-4, topology="scr")
rfc = RandomForestClassifier(n_estimators=300)

#pca = PCA()
pipe = Pipeline(steps=[('esn', esn), ('rfc', rfc)])

n_components = [20, 40, 64]
N = [10,50,100]
a = [0.01,0.1,0.5,1.0]
r = [0.01, 0.1, 1.0]
#Cs = np.logspace(-4, 4, 3)

max_depth = [5,10,15,20,50,100,200,500]

estimator = GridSearchCV(pipe,
                         dict(esn__N=N,
                              esn__a = a,
                              esn__r = r,
                              rfc__max_depth=max_depth), cv=5)
estimator.fit(Y_scaled, states)


## set up the grid search
#clf = GridSearchCV(pipeline, params, n_jobs=-1, verbose=1)
#clf.fit(Y_scaled, states)