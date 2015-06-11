
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.metrics import r2_score

class LinearFilter(BaseEstimator, TransformerMixin):

    def __init__(self, k=10, lamb=None):

        """
        Initialize hyperparameters

        :param k: window size
        :param lamb: regularization parameter
        :return:
        """

        self.k = k ## save window size k

        self.lamb = lamb ## save regularization parameter

        return



    def fit(self,X):
        """
        Fit method for the Echo State Network.

        :param X: data (K by D, where K is the number of data points, D the dimensionality)

        :return:
        """


        #return

    def transform(self, X):

        ## make sure data is in correct form (N_samples, N_dimensions)
        if len(X.shape) == 1:
            X = np.atleast_2d(X).T

        ## store data in attribute
        self.X = X

        ## number of data points
        self.K = int(self.X.shape[0])

        ## number of dimensions
        self.D = int(self.X.shape[1])


        ## filter windows
        H = np.zeros((self.K-self.k, self.k))

        for i in xrange(self.k,self.K-1,1):
            H[i-self.k,:] = X[i-self.k:i,0]


        self.H = H

        #print(self.k)
        if len(X.shape) == 1:
            X = np.atleast_2d(X).T

        H = self.H
        yy = X[self.k:]


        if self.lamb is None:
            ## proposals for regularization parameters
            lamb_all = [0.1, 1., 10.]
            ## initialize Ridge Regression classifier
            rr_clf = RidgeCV(alphas=lamb_all)
            ## fit the data with the linear model
            #print(H.shape)
            #print(yy.shape)
            rr_clf.fit(H, yy)
            ## regularization parameter determined by cross validation
            self.lamb = rr_clf.alpha_

        else:
            rr_clf = Ridge(alpha=self.lamb)
            rr_clf.fit(H,yy)

        ## best-fit output weights
        self.ww = rr_clf.coef_

        ## store activations for future use

        return self.ww


    def predict(self, X):
        """run the esn, see what happens"""

        ## data
        if len(X.shape) == 1:
            X = np.atleast_2d(X).T

        X_pred = []

        for i in range(self.k, X.shape[0]):
            hh = X[i-self.k:i,:]
            #print(hh.shape)
            X_pred.append(np.dot(self.ww, hh))

        return np.array(X_pred)[:,0,0]

    def score(self, X):
        """
        Score performance of the algorithm.

        :param X: (numpy.ndarray) data to predict

        :return:
        """
        if len(X.shape) == 1:
            X = np.atleast_2d(X).T

        ## train on the original data
        ww = self.transform(self.X)

        ## predict the new dataset given to the score method
        X_pred = self.predict(X)

        return r2_score(X, X_pred)

    def fit_transform(self, X, y=None):

        self.fit(X)
        ww = self.transform(X)

        return ww


class LinearFilterEnsemble(BaseEstimator, TransformerMixin):

    def __init__(self, k=100, lamb=None):

        """

        :param k:
        :return:
        """


        self.k = k
        self.lamb = lamb

        return


    def fit(self, X):
        return self

    def transform(self,X):

        ## number of samples per time series
        self.K = X.shape[1]

        ## number of time series
        self.L = X.shape[0]


        ensemble = []
        for l in xrange(self.L):
            lf = LinearFilter(k=self.k, lamb=self.lamb)
            lf.fit(X[l,:])
            ensemble.append(lf)

        self.ensemble = ensemble

        ww = np.zeros((X.shape[0], self.k))

        for l,e in zip(xrange(X.shape[0]), self.ensemble):
            ww[l,:] = e.transform(X[l,:])

        self.ww = ww
        return ww


    def fit_transform(self, X, y=None):

        self.fit(X)
        ww = self.transform(X)

        return ww

