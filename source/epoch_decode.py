
from myClass import mGaussian, ModelState
import math
import numpy as np

model_state_list = []

def lognormpdf(x,mu,S):
    """ Calculate gaussian probability density of x, when x ~ N(mu,sigma) """
    nx = len(S)
    norm_coeff = nx*math.log(2*math.pi)+np.linalg.slogdet(S)[1]
    err = x-mu
    numerator = np.linalg.solve(S, err).T.dot(err)
    return -0.5*(norm_coeff+numerator)

def model_state_transit_okay(m1, m2):
    if m1.epoch == m2.epoch and m1.model == m2.model:
        return(True)
    elif m1.epoch == m2.epoch-1 and m1.model != m2.model:
        return(True)
    else:
        return(False)

def init_state_list(k, n):
    """ We have k models and want to estimate n epochs """
    global model_state_list
    for i in xrange(k):
        for j in xrange(n):
            mState = ModelState(i,j)
            model_state_list.append(mState)
    
def decoding(pairs, kGaussians, n_epochs):
    """ Viterbi Decoding of Darwin Reading Epochs"""
    k_models = len(kGaussians)
    n_data = len(pairs)

    global model_state_list
    init_state_list(k_models, n_epochs)
    n_states = len(model_state_list)
    
    scores = np.array([[float('-inf') for x in xrange(n_data)] for y in xrange(n_states)])
    pred = np.array([[-1 for x in xrange(n_data)] for y in xrange(n_states)])
    
    print scores.shape
    
    print "----------------------------------"

    for i in xrange(n_states):
        if model_state_list[i].epoch == 0:
            scores[i][0] = 0

    for i in xrange(n_data):
        for j in xrange(n_states):
            if(scores[j][i] == float('-inf')):
                continue
            cur_state = model_state_list[j]
            cur_model = kGaussians[cur_state.model]
            cur_data = pairs[i,:]
            dat_likelihood = lognormpdf(cur_data,cur_model.mean,cur_model.sigma)
            scores[j][i] = scores[j][i] + dat_likelihood

            for z in xrange(n_states):
                next_state = model_state_list[z]
                if not model_state_transit_okay(cur_state, next_state):
                    continue

                if i + 1 >= n_data:
                    continue
            
                if scores[z][i+1] < scores[j][i]:
                    scores[z][i+1] = scores[j][i]
                    pred[z][i+1] = j


    #trace back
    best_state = -1
    i = n_data - 1
    temp_score = float('-inf')
    for j in xrange(n_states):
        if scores[j][i] > temp_score and model_state_list[j].epoch == n_epochs - 1:
            temp_score = scores[j][i]
            best_state = j

    print "best score", temp_score

    best_states = [best_state]
    print "trace back....."
    for i in reversed(range(1,n_data)):
        best_state = pred[best_state][i]
        best_states.insert(0, best_state)
    print best_states

def estimate_epochs(pairs, kGaussians):
    """wrapper: set the number of epochs to be the same with possible models"""
    decoding(pairs, kGaussians, len(kGaussians))
