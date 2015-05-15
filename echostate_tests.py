
### Tests for the echo state network code.

import numpy as np

from echostate import EchoStateNetwork


def test_hidden_weights():
    """
    Test the initialization for the hidden units
    :return:
    """

    ## choose value for r
    r = 0.1
    ## choose value for b, should be different from r
    ## for demonstration purposes
    b = 0.5

    ## pick 10 hidden units
    N = 10

    ## pick a weight for the input units
    a = 0.75

    #### go through the topologies

    ## SCR first
    esn = EchoStateNetwork(N=N,a=a,r=r,topology="scr")
    print("SCR topology: \n "
          "Elements should be on lower sub-diagonal only (U_(i+1,i)), plus one"
          "element on position in upper right corner: (0,N) \n" + str(esn.uu) + "\n")

    ## DLR second
    esn = EchoStateNetwork(N=N,a=a,r=r,topology="dlr")
    print("DLR topology: \n "
          "Elements should be on lower sub-diagonal only (U_(i+1,i)) \n" + str(esn.uu) + "\n")

    ## DLRB third
    esn = EchoStateNetwork(N=N,a=a,r=r,b=b,topology="dlrb")
    print("DLRB topology: \n "
          "Elements should be on lower sub-diagonal (U_(i+1,i)), "
          "on upper subdiagonal (U_(i,i+1)) \n" + str(esn.uu) + "\n")

    return


def test_input_weights():
    """
    Test the initialization of the weights of connections between input units and hidden units.
    Testing for both a single time series and multivariate data.
    :return:
    """
    ## dummy data
    x = np.arange(10)

    ## single data stream
    y = np.atleast_2d(np.ones_like(x)).T

    print(y.shape)
    ## double data stream
    yd = np.ones((len(x),2))
    print(yd.shape)

    ## choose value for r
    r = 0.1

    ## pick a weight for the input weights
    a = 0.75

    ## pick 10 hidden units
    N = 10


    ## single data stream
    esn = EchoStateNetwork(N=N,a=a,r=r,topology="scr")
    vv = esn._initialize_input_weights(y.shape[1],esn.a)
    print("Input weights for single data stream: \n " + str(vv))

    ## double data stream
    esn = EchoStateNetwork(N=N,a=a,r=r,topology="scr")
    vv = esn._initialize_input_weights(yd.shape[1],esn.a)
    print("Input weights for single data stream (should be randomly -0.75 and 0.75: \n " + str(vv))
    print("shape of data points: " + str(yd.shape))
    print("shape of input weights (should be the same as shape of data): " + str(vv.shape))

    return

def test_damping():

    ## choose value for r
    r = 0.1

    ## pick a weight for the input weights
    a = 0.75

    ## pick 10 hidden units
    N = 200

    ## number of iterations
    iterations = 200

    ## scaling for the memory
    scaling = 1.0

    esn = EchoStateNetwork(N=N,a=a,r=r,scaling=1.0,topology="scr")

    """Checking if the network stabilises"""
    """This is (modified) code from https://github.com/squeakus/echostatenetwork"""
    acts = np.zeros((iterations, esn.N))
    #set them to random initial activations
    acts[0,:] = np.random.uniform(-1,1,size=esn.N)

    # lets see if it dampens out
    for i in range(1, iterations):
        acts[i] = ((1-scaling) * acts[i-1])
        acts[i] += scaling * np.tanh(np.dot(acts[i-1],
                                         esn.uu))

    return acts


def test_code():

    data = np.loadtxt("varsine.dat")

    esn = EchoStateNetwork(N=100, a=1.0, r=0.5, n_washout=50)

    esn.fit(data)

    yy_test = esn.predict(data)

    return esn, esn.ww, yy_test


def test_with_noise():

    data = np.loadtxt("varsine.dat")

    yy = np.random.normal(data, 0.1)

    esn = EchoStateNetwork(N=100, a=1.0, r=0.5)
    esn.fit(yy)

    yy_test = esn.predict(yy)

    return esn, esn.ww, yy_test

def test_score():

    data = np.loadtxt("varsine.dat")

    yy = np.random.normal(data, 0.1)
    #print(yy.shape)

    esn = EchoStateNetwork(N=100, a=1.0, r=0.5)
    esn.fit(data)

    print("NSME should match sklearn.metrics.r2_score! \n")

    print("NSME score on full dataset without noise is: %.3f"%esn.score(data, method="nsme"))
    print("R^2 score on full dataset without noise is: %.3f \n"%esn.score(data, method="r2"))

    print("NSME score on full data set: train without noise, score with noise: %.3f"%esn.score(yy, method="nsme"))
    print("R^2score on full dataset without noise is: %.3f \n"%esn.score(yy, method="r2"))

    ### Split into training and test set, see what happens
    ndata = data.shape[0]

    ## fraction of data to use for training
    training_fraction = 0.6
    training_length = int(training_fraction*ndata)

    yy_train = yy[:training_length]
    yy_test = yy[training_length:]

    esn.fit(yy_train)

    print("NSME score on test dataset with noise is: %.3f"%esn.score(yy_test, method="nsme"))
    print("R^2 score on full dataset without noise is: %.3f"%esn.score(yy_test, method="r2"))

    return





### run all tests
test_hidden_weights()
test_input_weights()
acts = test_damping()
esn, ww, yy_test = test_code()
esn_noise, ww_noise, yy_test_noise = test_with_noise()
test_score()