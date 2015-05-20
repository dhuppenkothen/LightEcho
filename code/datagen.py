### GENERATE DATA FROM NARMA PROCESSES

import numpy as np

def narma1(nsamples):
    ## randomly generated inputs
    s = np.random.uniform(0.0,0.5, size=nsamples)

    ## set NARMA class
    narma_class = 9

    ## set up array for the outputs
    y = np.zeros(nsamples)
    ## loop through all elements
    for t in xrange(0,y.shape[0]-1):
        firstterm = 0.3*y[t]
        sumterm = np.sum([y[t-i] for i in xrange(narma_class + 1)])
        secondterm = 0.05*y[t]*sumterm
        thirdterm = 1.5*s[t-narma_class]*s[t]
        y[t+1] = firstterm + secondterm + thirdterm + 0.1
    return y

def narma2(nsamples):
    s = np.random.uniform(0.0, 0.5, size=nsamples)
    y = np.zeros(nsamples)
    #print(y.shape)
    narma_class = 19
    for t in xrange(0,y.shape[0]-1):
        firstterm = 0.3*y[t]
        sumterm = np.sum([y[t-i] for i in xrange(narma_class+1)])
        second_term = 0.05*y[t]*sumterm
        thirdterm = 1.5*s[t-narma_class]*s[t]
        tanhterm = np.tanh(firstterm + second_term + thirdterm + 0.01)
        y[t+1] = tanhterm + 0.2
    return y

def narma3(nsamples):
    s = np.random.uniform(0.0, 0.5, size=nsamples)
    y = np.zeros(nsamples)
    narma_class = 29
    for t in xrange(0,y.shape[0]-1):
        firstterm  = 0.2*y[t]
        sumterm = np.sum([y[t-i] for i in xrange(narma_class+1)])
        secondterm = 0.004*s[t]*sumterm
        thirdterm = 1.5*s[t-narma_class]*s[t] + 0.201
        y[t+1] = firstterm + secondterm + thirdterm
    return y