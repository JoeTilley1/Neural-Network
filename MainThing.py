__author__ = 'Joe'
import numpy as np
import random

class Network(object):

    def __init__(self, sizes):
        self.sizes = sizes
        self.L = len(sizes)
        self.biases = [np.random.normal(0, 1, (sizes[n+1], 1)) for n in range(len(sizes)-1)]
        self.weights = [np.random.normal(0, 1, (sizes[n+1], sizes[n])) for n in range(len(sizes)-1)]

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
            delta = [np.multiply(dCdx_L(x_L, y),sigmoid_prime(z[-1]))]
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

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data):
        '''Completes the method of stochastic gradient descent'''
        for n in range(epochs):
            mini_batch = random.sample(training_data, mini_batch_size)
            [dCdb, dCdw] = self.backpropogation(mini_batch)
            self.biases = [np.add(b,np.multiply(-(eta/mini_batch_size),Cb)) for b, Cb in zip(self.biases, dCdb)]
            self.weights = [np.add(w,np.multiply(-(eta/mini_batch_size),Cw)) for w, Cw in zip(self.weights, dCdw)]
            print("Epoch {0}: {1} / {2}".format(n, self.evaluate(test_data), len(test_data)))

    def evaluate(self, test_data):
        '''Calculates number of correct answers network produces for the test data'''
        correct = 0
        for (x,y) in test_data:
            network_ans = self.feedforward(x)[1]
            real_ans = y
            if network_ans.index(max(network_ans)) == real_ans.index(max(real_ans)):
                xy_works = 1
            else:
                xy_works = 0
            correct += xy_works
        return correct

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_prime(x):
    return np.divide(np.exp(x),((np.exp(x)+1)**2))

def dCdx_L(x, y):
    ''' Calculates derivatives wrt x_L of C.
    x is column vector of activation outputs, x_L.
    y is column vector of true outputs.'''
    return np.add(x,-y)


net = Network([2,4,3,1])
training_data = []
training_data.append((np.array([[1],[2]]),np.array([[1]])))
print(net.feedforward(training_data[0][0])[1])

net.SGD(training_data, 100, 1, 1, training_data)