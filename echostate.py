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

def sigmoid(x):
    return 1./(1.+np.exp(-x))



class EchoStateNetwork(object):


    def __init__(self, x,y, N, lamb, a, r, b=None,topology="scr"):
        """
        Initialization for the echo state network for time series.
        :param x: x-coordinate (time bins)
        :param y: data (K by D, where K is the number of data points, D the dimensionality)
        :param N: number of hidden units.
        :param lamb: regularization term
        :param a: absolute value of input weights
        :param r: weights for forward connections between hidden weights
        :param b: weights for backward connections between hidden weights (for topology="dlrb" only)
        :param topology: reservoir topology (one of "SCR", "DLR", "DLRB",
                see Rodan+Tino, "Minimum Complexity Echo State Network" for details
        :return:
        """


        ## x-coordinate
        self.x = x

        ## y-coordiante
        self.y = y
        print("shape of data stream: " + str(self.y.shape))

        ## number of data points
        self.K = self.y.shape[0]
        print("Number of data points: %i"%self.K)

        ## number of dimensions
        if len(self.y.shape) > 1:
            self.D = self.y.shape[1]
        else:
            self.D = 1

        print("Dimensionality of the data: %i"%self.D)

        ## number of hidden units
        self.N = N
        print("Number of hidden units: %i"%self.N)

        ## output weight matrix
        self.ww = np.zeros((self.N, self.D))

        ## regularisation term
        self.lamb = lamb

        ## reservoir topology and associated parameters
        self.topology = topology
        self.r = r
        self.b = b

        ## input unit weights
        self.a = a

        ## initialize input weights
        self.vv = self._initialize_input_weights(self.a)

        ## initialize hidden weights
        self.uu = self._initialize_hidden_weights(self.r,self.b,self.topology)


    def _initialize_input_weights(self, a):
        """
        Initialize input weights.
        Input layer fully connected to reservoir, weights have same absolute value a,
        but signs randomly flipped for each weight.

        :param a: weight value
        :return: vv = input layer weights
        """

        ## probability for the Bernoulli trials
        pr = 0.5

        ## set the seed deterministically
        np.random.seed(seed=20150513)

        ## initialize weight matrix with Bernoulli distribution
        vv = scipy.stats.bernoulli.rvs(pr, size=(self.N, self.D)).astype(np.float64)

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

    def train(self, ww_init, n_washout=100):
        """
        not sure I'm doing the right thing in this method!!!!

        :param ww_init: initial weights to use
        :param n_washout: number of samples in the time series to wash out
        :return:
        """

        ## not sure this is right?
        X = np.ones((self.N, 1))

        for i in xrange(self.K-1):
            X[i+1] = np.tanh(np.dot(self.uu, X[i]) + np.dot(self.vv,self.yy))

        ## discard washout period
        X = X[n_washout:]
        #vv = self.vv[n_washout:]
        ww = self.ww[n_washout:]
        ww_init = ww_init[n_washout:]
        yy = self.y[n_washout:]


    def _cost_function(self, yy, X, ww, lamb):

        inner_part =  np.dot(X,ww) - yy
        second_term = 0.5*lamb*np.linalg.norm(ww)**2.
        return 0.5*np.linalg.norm(inner_part)**2. + second_term




        return
