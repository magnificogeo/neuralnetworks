import numpy as np


## ---------------- Part 1 -----------------------##


# This is a supervised learning algorithm
# X = (hours sleeping, hours studying), y = Score on test
X = np.array(([3,5],[5,1],[10,2]), dtype=float)
y = np.array(([75],[82],[93]), dtype=float)

# Normalizing the input
X = X/np.amax(X, axis=0)
y = y/100 #MAx test score is 100

## ---------------- Part 2 -----------------------##

# Input is 2 and output is 1 neuron
# We will use 1 hidden layer with 3 neurons

class Neural_Network(object):

    def __init__(self):
        #define Hyperparameters
        self.inputLayerSize = 2
        self.outputLayerSize = 1
        self.hiddenLayerSize = 3

        #weights definition


        # input X is a 3-by-2 matrix
        # 2-by-3 matrix (2 rows and 3 columns)
        self.W1 = np.random.randn(self.inputLayerSize, self.hiddenLayerSize)

        # 3-by-1 matrix (3 rows and 1 column)
        self.W2 = np.random.randn(self.hiddenLayerSize, self.outputLayerSize)

    # X is an input in the forward loop
    def forward(self, X):
        #Propagating inputs through the network
        #marix dot product
        self.z2 = np.dot(X, self.W1)

        #a2 is the output of the first hidden layer after applying sigmoid
        # activation function
        self.a2 = self.sigmoid(self.z2)

        # matrix multiplication of the final layer
        self.z3 = np.dot(self.a2, self.W2)

        # yHat is final output
        yHat = self.sigmoid(self.z3)
        return yHat

    def sigmoid(self, z):
        #Applying sigmoid activation function to scalar
        return 1/(1+np.exp(-z))

    # this is the derivative of the sigmoid function
    def sigmoidPrime(self, z):
        return np.exp(-z)/((1+np.exp(-z))**2)


    def costFunction(self, X, y):
        #Compute cost for given X,y use weights already stored in class
        self.yHat = self.forward(X)
        # J is the cost function
        J = 0.5 * sum((y - self.yHat)**2)
        return J


    def costFunctionPrime(self, X, y):
        # Compute derivative with respect to W and W2 for a given X and y:

        self.yHat = self.forward(X)

        delta3 = np.multiply(-(y-self.yHat), self.sigmoidPrime(self.z3)
        dJdW2 = np.dot(self.a2.T, delta3)

        delta2 = np.dot(delta3, self.W2.T)*self.sigmoidPrime(self.z2)
        dJdW1 = np.dot(X.T, delta2)


        return dJdW1, dJdW2
