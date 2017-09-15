__author__ = 'Joe'

import numpy as np
from numpy import linalg as LA
import random

'''Default set to cross entropy function with L2 regularization and modified weight initialization'''

class Network(object):

    def __init__(self, sizes):
        self.sizes = sizes
        self.L = len(sizes)
        self.biases = [np.random.normal(0, 0, (sizes[n+1], 1)) for n in range(len(sizes)-1)]
        self.weights = [np.random.normal(0, np.divide(1,np.sqrt(sizes[n])), (sizes[n+1], sizes[n])) for n in range(len(sizes)-1)]

    def feedforward(self, input_vector):
        ''' Runs the input vector through the network'''
        z = [np.add(self.weights[0].dot(input_vector), self.biases[0])]
        for n in range(1,self.L-1):
            z.append(np.add(self.weights[n].dot(sigmoid(z[-1])), self.biases[n]))
        output_vector = sigmoid(z[-1])
        return [z,output_vector]

    def backpropogation(self, mini_batch):
        '''Calculates sum of partial derivatives of C wrt weights and biases for a given minibatch'''
        dCdb = [np.zeros(b.shape) for b in self.biases]
        dCdw = [np.zeros(w.shape) for w in self.weights]
        for (x,y) in mini_batch:
            [z,x_L] = self.feedforward(x)
            activations = [sigmoid(zed) for zed in z]
            activations.insert(0,x)
            delta = [x_L - y] #Computationally cheap method for cross entropy
            for n in range(self.L-2,0,-1):
                delta.insert(0,np.multiply(((self.weights[n]).transpose()).dot(delta[0]),sigmoid_prime(z[n-1])))
            db = delta
            dw = [np.zeros(w.shape) for w in self.weights]
            for l in range(self.L - 1):
                for i in range(self.sizes[l+1]):
                    for j in range(self.sizes[l]):
                        dw[l][i][j] = delta[l][i]*activations[l][j]
            dCdb = [np.add(Cb,b) for Cb, b in zip(dCdb, db)]
            dCdw = [np.add(Cw,w) for Cw, w in zip(dCdw, dw)]
        return [dCdb, dCdw]

    def SGD(self, training_data, epochs, mini_batch_size, eta, lam, test_data):
        '''Completes the method of stochastic gradient descent'''
        for n in range(epochs):
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)] # Method modified to mock Nielsen's learning pace
            for mini_batch in mini_batches:
                [dCdb, dCdw] = self.backpropogation(mini_batch)
                self.biases = [np.add(b,np.multiply(-(eta/mini_batch_size),Cb)) for b, Cb in zip(self.biases, dCdb)]
                self.weights =  [np.add(np.multiply(1-lam*eta/len(test_data),w),np.multiply(-(eta/mini_batch_size),Cw)) for w, Cw in zip(self.weights, dCdw)]
            print("Epoch {0}: {1} / {2}".format(n, self.evaluate(test_data)[0], len(test_data), self.evaluate(test_data)[1]))


    def evaluate(self, test_data):
        '''Calculates number of correct answers network produces for the test data'''
        correct = 0
        thecost = 0
        for (x,y) in test_data:
            network_ans = self.feedforward(x)[1]
            real_ans = y
            if np.argmax(network_ans) == np.argmax(real_ans):
                xy_works = 1
            else:
                xy_works = 0
            correct += xy_works
            thecost += cost(network_ans, real_ans)
        return [correct,thecost]

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_prime(x):
    return np.exp(x)/(np.exp(x)+1)**2

def dCdx_L(x, y):
    ''' Calculates derivatives wrt x_L of C.
	x is column vector of activation outputs, x_L.
	y is column vector of true outputs.'''
    return np.add(x,-y)

def cost(a, y):
    '''Cross entropy cost'''
    return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

import csv
with open('voice.csv', 'rt') as f:
	reader = csv.reader(f)
	all_data = [row for row in reader]
	inputs = []
	outputs = []
	for i in range(len(all_data)):
		inputs.append(np.transpose(np.array([[float(j) for j in all_data[i][:-2]]])))
		outputs.append(np.transpose(np.array([[float(j) for j in all_data[i][-2:]]])))
	data = [(inp, out) for inp, out in zip(inputs, outputs)]

validation_data = data[0:350]
test_data = data[350:700]
training_data = data[700:]

net = Network([20,2])
net.SGD(training_data, 100, 1, 0.001, 0, validation_data)