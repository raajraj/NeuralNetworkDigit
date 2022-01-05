import csv
import sys
import math
import warnings
import numpy as np
from numpy import exp

np.random.seed(0)
warnings.filterwarnings('ignore')

# take in command line argument files
parameter1 = open(str(sys.argv[1]))
parameter2 = open(str(sys.argv[2]))
#paramter3 = open(str(sys.argv[3]))

type(parameter1)
type(parameter2)
#type(paramter3)

# extract csv files
extract_training_set = csv.reader(parameter1)
extract_training_labels = csv.reader(parameter2)
#extract_test_images = csv.reader(paramter3)

# make into arrays
training_data_list = []
for row in extract_training_set:
    training_data_list.append(row)
training_data = np.asarray(training_data_list, dtype=np.int_, order='C')

training_labels_list = []
for row in extract_training_labels:
    training_labels_list.append(row)
training_labels = np.asarray(training_labels_list, dtype=np.int_, order='C')

#test_images_list = []
#for row in extract_test_images:
 #   test_images_list.append(row)
#test_images = np.asarray(test_images_list, dtype=np.int_, order='C')


# putting into batches
data_points = len(training_data)
x = 0
y = 50
batches = []
for i in range(1200):
    batch = training_data[x:y, 0:784]
    batch = batch/batch.max()
    batches.append(batch)
    x = x + 50
    y = y + 50

x = 0
y = 50
labels = []
for i in range(1200):
    label = training_labels[x:y]
    labels.append(label)
    x = x + 50
    y = y + 50

# Neural Network
class NeuralNetwork(object):
    def __init__(self):
        self.lamb = 1e-3
        self.W1 = 0.01*np.random.randn(784, 300)
        self.W2 = 0.01*np.random.randn(300, 16)
        self.W3 = 0.01*np.random.randn(16, 10)
        self.biases1 = np.zeros((1,300))
        self.biases2 = np.zeros((1,16))
        self.biases3 = np.zeros((1,10))

    # sigmoid derivative function
    def sigmoid_derivative(self,x):
        s = self.sigmoid(x)
        ds = s*(1-s)
        return ds

    # sigmoid activation function
    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))

    # softmax activation function
    def softmax(self, x):
        for y in x:
            y -= np.max(y)
        exp_scores = np.exp(x)
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    
    # ReLu activation function
    def rectified(self, x):
        return np.maximum(0.0, x)
    
    # loss function
    def compute_loss(self, X, output, Y):
        num_examples = X.shape[0]
        neg_log = -np.log(output[range(num_examples), Y])
        data_loss = np.mean(neg_log)
        reg_loss = 0.5*self.lamb*np.sum(self.W3*self.W3) + 0.5*self.lamb*np.sum(self.W2*self.W2) + 0.5*self.lamb*np.sum(self.W1*self.W1)
        loss = data_loss + reg_loss
        print(loss)
        return loss
    
    def compute_gradient(self, X, output, Y):
        num_examples = X.shape[0]
        g = output
        for i in range(num_examples):
            g[i, Y[i]] -= 1
        g /= num_examples
        return g

    # forward propagation
    def forward_propagation(self,X):
        self.layer1 = np.dot(X, self.W1) + self.biases1
        self.z = self.rectified(self.layer1)
        self.layer2 = np.dot(self.z, self.W2) + self.biases2
        self.z2 = self.rectified(self.layer2)
        self.layer3 = np.dot(self.z2, self.W3) + self.biases3
        output = self.softmax(self.layer3)
        return output

    # backward propagation
    def backward_propagation(self, X, y, output, g):
        alpha = -0.1
        
        dW3 = alpha * np.dot(self.z2.T, g)
        db3 = alpha * np.sum(g, axis=0, keepdims=True)
        dhidden2 = np.dot(g, self.W3.T) * self.sigmoid_derivative(self.z2)
        dhidden2[self.z2 <= 0] = 0
        
        dW2 = alpha * np.dot(self.z.T, dhidden2)
        db2 = alpha * np.sum(dhidden2, axis=0, keepdims=True)
        dhidden = np.dot(dhidden2, self.W2.T) * self.sigmoid_derivative(self.z)
        dhidden[self.z <= 0] = 0
        
        dW = alpha * np.dot(X.T, dhidden)
        db = alpha * np.sum(dhidden, axis=0, keepdims=True)
        
        self.W1 += dW
        self.biases1 += db
        self.W2 += dW2
        self.biases2 += db2
        self.W3 += dW3
        self.biases3 += db3
        return

    def train(self, X, Y):
        output = self.forward_propagation(X)
        loss = self.compute_loss(X, output, Y)
        gradient = self.compute_gradient(X, output, Y)
        self.backward_propagation(X, Y, output, gradient)
        return

NN = NeuralNetwork()

for batch, Y in zip(batches, labels):
    for i in range(20):
        NN.train(batch, Y)
    #for b,y in zip(batch,Y):
        #arr = NN.forward_propagation(b)
        #result = arr[0]
        #maxi = np.amax(result)
        #out = str([np.where(result == maxi)])
        #print("Predicted Output: " + out[9])
        #print("Actual Output: " + str(y))

#with open('test_predictions.csv', mode='w') as csv_file:
   # to = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
   # for t in test_images:
       # arr = NN.forward_propagation(t)
       # result = arr[0]
       # maxi = np.amax(result)
       # out = str([np.where(result == maxi)])
       # to.writerow(out[9])
