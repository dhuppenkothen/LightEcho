
### Tests for the echo state network code.

import numpy as np
from echostate import EchoStateNetwork

def test_hidden_weights():
    """
    Test the initialization for the hidden units
    :return:
    """
    ## dummy data
    x = np.arange(10)
    y = np.ones_like(x)

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
    esn = EchoStateNetwork(x,y,N,a,r, b,topology="scr")
    print("SCR topology: \n "
          "Elements should be on lower sub-diagonal only (U_(i+1,i)), plus one"
          "element on position in upper right corner: (0,N) \n" + str(esn.uu) + "\n")

    ## DLR second
    esn = EchoStateNetwork(x,y,N,a,r,b,topology="dlr")
    print("DLR topology: \n "
          "Elements should be on lower sub-diagonal only (U_(i+1,i)) \n" + str(esn.uu) + "\n")

    ## DLRB third
    esn = EchoStateNetwork(x,y,N,a,r,b,topology="dlrb")
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
    y = np.ones_like(x)

    ## double data stream
    yd = np.ones((len(x), 2))
    print(yd.shape)

    ## choose value for r
    r = 0.1

    ## pick a weight for the input weights
    a = 0.75

    ## pick 10 hidden units
    N = 10


    ## single data stream
    esn = EchoStateNetwork(x,y,N,a,r,topology="scr")
    print("Input weights for single data stream: \n " + str(esn.vv))

    ## double data stream
    esn = EchoStateNetwork(x,yd,N,a,r,topology="scr")
    print("Input weights for single data stream (should be randomly -0.75 and 0.75: \n " + str(esn.vv))
    print("shape of data points: " + str(esn.y.shape))
    print("shape of input weights (should be the same as shape of data): " + str(esn.vv.shape))

    return


def test_cost_function():


    ## dummy data
    x = np.arange(10000)

    ## single data stream
    y = np.ones_like(x)


    ## choose value for r
    r = 0.1

    ## pick a weight for the input weights
    a = 0.75

    ## pick 10 hidden units
    N = 15

    ## set regularisation term
    lamb = 1.0

    esn = EchoStateNetwork(x,y,N,a,r,topology="scr")

    weights = np.random.uniform(size=(esn.N, esn.D))

    ## not sure this is right?
    X = np.ones((esn.K, esn.N))

    for i in xrange(esn.K-1):
        first_part = np.dot(esn.uu, X[i])
        #print("first part: " + str(first_part.shape))
        second_part = np.dot(esn.vv,esn.y.T)
        #print("second part: " + str(second_part.shape))
        X[i+1,:] = np.tanh(first_part, second_part.T)
        #X[i+1] = np.tanh(np.dot(esn.uu, X[i]) + np.dot(esn.vv,esn.y))


    cf = esn._cost_function(y, X, weights, lamb)
    print("The cost function is: " + str(cf))

    return

def test_damping():

    ## dummy data
    x = np.arange(10000)

    ## single data stream
    y = np.ones_like(x)


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

    esn = EchoStateNetwork(x,y,N,a,r,topology="scr")

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
    x = np.arange(len(data))

    esn = EchoStateNetwork(x, data, 100, 1.0, 0.5)

    ww, yy_out = esn.train(n_washout=50, scaling=1.0)

    yy_test = esn.test(data, scaling=1.0)

    return ww, yy_out, yy_test


### run all tests
#test_hidden_weights()
#test_input_weights()
#est_cost_function()
#acts = test_damping()
ww, yy_out, yy_test = test_code()