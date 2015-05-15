### This is an attempt at making an echo state network work"

## Some naming conventions (following Giannotis+ 2015)
## y: time series points at time t
## t: times at which the time series is measured
## N: number of hidden units
## v: input weights, of dimension Nx1
## x: hidden units, of length N (could be changed?)
## u: weights of the reservoir units (vector of shape NxN)
## w: readout weights, vector of shape NX1

import numpy as np
import scipy.stats
from sklearn.linear_model import RidgeCV, Ridge

def sigmoid(x):
    return 1./(1.+np.exp(-x))



class EchoStateNetwork(object):


    def __init__(self, N = 100, a = 0.75, r = 0.1, n_washout=100,
                 scaling=1.0, lamb=None, seed=20150513, b=None,topology="scr"):
        """
        Initialization for the echo state network for time series.
        :param N: number of hidden units.
        :param lamb: regularization term
        :param a: absolute value of input weights
        :param r: weights for forward connections between hidden weights
        :param n_washout: number of samples in the time series to wash out
        :param scaling: scaling between current and previous time step
        :param lamb: regularization parameter for ridge regression
        :param seed: seed for the random number generator
        :param b: weights for backward connections between hidden weights (for topology="dlrb" only)
        :param topology: reservoir topology (one of "SCR", "DLR", "DLRB",
                see Rodan+Tino, "Minimum Complexity Echo State Network" for details
        :return:
        """



        ## number of hidden units
        self.N = int(N)
        print("Number of hidden units: %i"%self.N)

        ## reservoir topology and associated parameters
        self.topology = topology
        self.r = r
        self.b = b

        ## input unit weights
        self.a = a


        ## number of "washout" samples
        self.n_washout = n_washout

        ## Scaling between current and pervious time step
        self.scaling = scaling

        ## Regularization parameter
        self.lamb = lamb

        ## seed for the random number generator
        self.seed = seed

        ## initialize hidden weights
        self.uu = self._initialize_hidden_weights(self.r,self.b,self.topology)


    def _initialize_input_weights(self, D, a, seed=20150513):
        """
        Initialize input weights.
        Input layer fully connected to reservoir, weights have same absolute value a,
        but signs randomly flipped for each weight.

        :param D: dimensionality of input data
        :param a: weight value
        :return: vv = input layer weights
        """

        ## probability for the Bernoulli trials
        pr = 0.5

        ## set the seed deterministically
        np.random.seed(seed=seed)

        ## initialize weight matrix with Bernoulli distribution
        vv = scipy.stats.bernoulli.rvs(pr, size=(self.N, D)).astype(np.float64)

        ## populate with actual weights
        vv[vv == 0.] = -a
        vv[vv == 1.] = a

        return vv

    def _initialize_hidden_weights(self,r, b=None, type="scr"):
        """
        Initialize the weights for the connections between the
        hidden units.
        Options for the typology of the reservoir are:
            - SCR (Simple Cycle Reservoir): units organised in a a cycle
            - DLR (Delay Line Reservoir): units organised in a line
            - DLRB (Delay Line Reservoice /w backward connection)
        :param r: weight value
        :param b: weight value for backward connections (for DLRB)
        :param type: string; "scr", "dlr", "dlrb", "esn"

        :return: uu = hidden weights
        """

        ## if we're using a DLRB topology, b needs to have a value!
        if type == "dlrb":
            assert(b is not None)

        ## initialize the array to store the weights in
        uu = np.zeros((self.N, self.N))

        ## all three topologies have the lower subdiagonal filled
        for i in xrange(self.N):
            for j in xrange(self.N):
                if i == j+1:
                    uu[i,j] = r

        ## if DLRB topology, fill upper subdiagonal with backwards connections
        if type == "dlrb":
            for i in xrange(self.N):
                for j in xrange(self.N):
                    if i+1 == j:
                        uu[i,j] = b

        ## if SCR, fill the connection between the last and the first node
        ## to make the line into a circle
        if type == "scr":
            uu[0,-1] = r

        return uu

    def fit(self,X):
        """
        Fit method for the Echo State Network.

        :param X: data (D by K, where K is the number of data points, D the dimensionality)

        :return:
        """



        ## data
        self.X = np.atleast_2d(X)
        print("shape of data stream: " + str(self.X.shape))

        ## number of data points
        self.K = int(self.X.shape[1])
        print("Number of data points: %i"%self.K)

        ## number of dimensions
        if len(self.X.shape) > 1:
            self.D = int(self.X.shape[0])
        else:
            self.D = 1

        print("Dimensionality of the data: %i"%self.D)


        ## initialize input weights
        self.vv = self._initialize_input_weights(self.D,self.a)


        ## initialize hidden units
        H = np.zeros((self.K, self.N))

        ## set first row with random numbers between -1 and 1
        H[0,:] = np.random.uniform(-1,1,size=(self.N))

        ## compute activations for every time step
        ## include a scaling for memory from previous time step
        for i in xrange(self.K-1):
            ## split equation into two parts to avoid stupid mistakes
            first_part = np.dot(H[i,:],self.uu)
            second_part = np.dot(self.vv,self.X[0,i]).T
            act = np.tanh(first_part + second_part)
            H[i+1,:] = (1.0-self.scaling)*H[i,:] + self.scaling*act


        ## discard washout period
        H = H[self.n_washout:,:]
        print("H.shape: " + str(H.shape))

        yy = self.X[:,self.n_washout:].T
        print("yy.shape: " + str(yy.shape))


        ## if regularization parameter is None, then determine by cross validation
        if self.lamb is None:
            ## proposals for regularization parameters
            lamb_all = [0.1, 1., 10.]
            ## initialize Ridge Regression classifier
            rr_clf = RidgeCV(alphas=lamb_all)
            ## fit the data with the linear model
            rr_clf.fit(H, yy)
            ## regularization parameter determined by cross validation
            self.lamb = rr_clf.alpha_

        else:
            rr_clf = Ridge(alpha=self.lamb)
            rr_clf.fit(H,yy)

        ## best-fit output weights
        self.ww = rr_clf.coef_

        ## store activations for future use
        self.H = H

        return

    def predict(self, X):
        """run the esn, see what happens"""


        H = np.zeros((X.shape[1], self.N))
        H[0,:] = self.H[-1,:]

        X_pred = []

        for i in range(X.shape[1]-1):

            X_pred.append(np.dot(H[i,:], self.ww.T))

            first_part = np.dot(H[i,:],self.uu)
            #print("first part: " + str(first_part.shape))
            second_part = np.dot(self.vv, X[0,i]).T
            #print("second part: " + str(second_part.shape))
            act = np.tanh(first_part + second_part)
            H[i+1,:] = (1.0-self.scaling)*H[i,:] + self.scaling*act

        return np.array(X_pred)

    def _cost_function(self, X, H, ww, lamb):
        """
        This is the cost function for Ridge Regression. I can just use the
        scikit-learn implementation of Ridge Regression, but with sqrt(lamb)
        as a regularization parameter.


        :param yy:
        :param X:
        :param ww:
        :param lamb:
        :return:
        """
        inner_part =  np.dot(H,ww) - X
        second_term = 0.5*lamb*np.linalg.norm(ww)**2.
        return 0.5*np.linalg.norm(inner_part)**2. + second_term


    def score(self, X, method="nsme"):
        """
        Score performance of the algorithm.

        :param X: (numpy.ndarray) data to predict
        :param method: (string) which scoring method to use:
                        nsme = Normalized Mean Square Error (Rodan+Tino 2011)
        :return:
        """
        X_pred = self.predict(X)


        if method == "nsme":
            nominator = np.mean(np.linalg.norm(X_pred - X, axis=0)**2.)
            denominator = np.mean(np.linalg.norm(X - np.mean(X, axis=1), axis=0)**2.)
            return nominator/denominator
        else:
            raise Exception("Scoring method not known!")
