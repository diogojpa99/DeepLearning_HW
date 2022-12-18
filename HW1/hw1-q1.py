#!/usr/bin/env python

# Deep Learning Homework 1

import argparse
import random
import os

import numpy as np
import matplotlib.pyplot as plt

import utils

def configure_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

def softmax(z):
    """
    Softmax function.
    Implemented by the group.
    """
    exp_z = np.exp(z - np.max(z))
    return exp_z / np.sum(exp_z)

def relu(z):
    """
    RELU activation function for mlp with one hidden layer.
    Implemented by the group.
    """
    return np.maximum(0,z)

def relu_prime(z):
    """
    Derivative of the RELU activation function for the whole array.
    Implemented by the group.
    """
    z[np.where(z>0)] = 1
    z[np.where(z<=0)] = 0
    return z

class LinearModel(object):
    def __init__(self, n_classes, n_features, **kwargs):
        # Weights are initialized at zeros
        self.W = np.zeros((n_classes, n_features))

    def update_weight(self, x_i, y_i, **kwargs):
        # We need to implement 
        raise NotImplementedError

    def train_epoch(self, X, y, **kwargs):
        for x_i, y_i in zip(X, y):
            self.update_weight(x_i, y_i, **kwargs)

    def predict(self, X):
        """X (n_examples x n_features)"""
        scores = np.dot(self.W, X.T)  # (n_classes x n_examples)
        predicted_labels = scores.argmax(axis=0)  # (n_examples)
        return predicted_labels

    def evaluate(self, X, y):
        """
        X (n_examples x n_features):
        y (n_examples): gold labels
        """
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible


class Perceptron(LinearModel):
    def update_weight(self, x_i, y_i, **kwargs):
        """
        x_i (n_features): a single training example
        y_i (scalar): the gold label for that example
        other arguments are ignored
        """
        # Q1.1a
        # (1) Prediction
        y_hat = self.predict(x_i)
        
        # (2) Update weigths
        if y_hat != y_i: 
            self.W[y_i] += x_i #increase weight of gold class 
            self.W[y_hat] -= x_i #decrease weight of incorrect class
        

class LogisticRegression(LinearModel):
    def update_weight(self, x_i, y_i, learning_rate=0.001):
        """
        x_i (n_features): a single training example
        y_i: the gold label for that example
        learning_rate (float): keep it at the default value for your plots
        """        
        # Q1.1b
        # (Init) Reshap input 
        x_i = x_i.reshape(np.size(x_i,0),1)
        
        # (1) Class scores
        class_scores = self.W.dot(x_i)
        
        # (2) Class probabilities 
        class_probs = softmax(class_scores)
        
        # (3) One-hot encoding of the output vector
        y_one_hot = np.zeros((np.size(self.W,0),1))  
        y_one_hot[y_i] = 1   
        
        # (4) SGD update
        self.W += learning_rate * np.outer((y_one_hot - class_probs), x_i.T)


class MLP(object):
    # Q3.2b. This MLP skeleton code allows the MLP to be used in place of the
    # linear models with no changes to the training loop or evaluation code
    # in main().
    def __init__(self, n_classes, n_features, hidden_size, layers):
        # Initialize an MLP with a single hidden layer.
        
        # (1) Initialize bias with zero vectors
        self.b1 = np.zeros((hidden_size,1))
        self.b2 = np.zeros((n_classes,1))
        
        # (2) Initialize weights matrices with random values
        self.W1 = np.random.normal(loc=0.1, scale=0.1, size=(hidden_size,n_features))
        self.W2 = np.random.normal(loc=0.1, scale=0.1, size=(n_classes,hidden_size))

    def predict(self, X):
        # Compute the forward pass of the network. At prediction time, there is
        # no need to save the values of hidden nodes, whereas this is required
        # at training time.
        
        # Simply use the feedforward step for the output
        z1 = self.W1.dot(X.T) + self.b1
        h1 = relu(z1)
        z2 = self.W2.dot(h1) + self.b2
        h2 = softmax(z2)
        
        return np.argmax(h2, axis = 0)

    def evaluate(self, X, y):
        """
        X (n_examples x n_features)
        y (n_examples): gold labels
        """
        # Identical to LinearModel.evaluate()
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible

    def train_epoch(self, X, y, learning_rate=0.001):
        # For each instance
        for x_i,y_i in zip(X,y):
            
            # (Init) Reshaping the input:
            x_i = x_i.reshape(np.size(x_i,0),1)
            
            # (1) One-hot encoding of the output vector
            y_one_hot = np.zeros((np.size(self.W2,0),1))  
            y_one_hot[y_i] = 1   
            
            # (2) Feedfoward propagation step
            z1 = self.W1.dot(x_i) + self.b1
            h1 = relu(z1)
            z2 = self.W2.dot(h1) + self.b2
            h2 = softmax(z2)
            
            # (3) Back propagation step
            z2_grad = h2 - y_one_hot
            w2_grad = z2_grad.dot(h1.T)
            b2_grad = z2_grad
            z1_grad = self.W2.T.dot(z2_grad) * relu_prime(z1)
            w1_grad = z1_grad.dot(x_i.T)
            b1_grad = z1_grad
            
            # (4) Updating the weights using SGD
            self.W1 -= learning_rate * w1_grad
            self.b1 -= learning_rate * b1_grad
            self.W2 -= learning_rate * w2_grad
            self.b2 -= learning_rate * b2_grad

            

def plot(epochs, valid_accs, test_accs):
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.xticks(epochs)
    plt.plot(epochs, valid_accs, label='validation')
    plt.plot(epochs, test_accs, label='test')
    plt.legend()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model',
                        choices=['perceptron', 'logistic_regression', 'mlp'],
                        help="Which model should the script run?")
    parser.add_argument('-epochs', default=20, type=int,
                        help="""Number of epochs to train for. You should not
                        need to change this value for your plots.""")
    parser.add_argument('-hidden_size', type=int, default=200,
                        help="""Number of units in hidden layers (needed only
                        for MLP, not perceptron or logistic regression)""")
    parser.add_argument('-layers', type=int, default=1,
                        help="""Number of hidden layers (needed only for MLP,
                        not perceptron or logistic regression)""")
    parser.add_argument('-learning_rate', type=float, default=0.001,
                        help="""Learning rate for parameter updates (needed for
                        logistic regression and MLP, but not perceptron)""")
    opt = parser.parse_args()

    utils.configure_seed(seed=42)

    add_bias = opt.model != "mlp"
    data = utils.load_classification_data(bias=add_bias)
    train_X, train_y = data["train"]
    dev_X, dev_y = data["dev"]
    test_X, test_y = data["test"]

    n_classes = np.unique(train_y).size  # 10
    n_feats = train_X.shape[1]

    # initialize the model
    if opt.model == 'perceptron':
        model = Perceptron(n_classes, n_feats)
    elif opt.model == 'logistic_regression':
        model = LogisticRegression(n_classes, n_feats)
    else:
        model = MLP(n_classes, n_feats, opt.hidden_size, opt.layers)
    epochs = np.arange(1, opt.epochs + 1)
    valid_accs = []
    test_accs = []
    for i in epochs:
        print('Training epoch {}'.format(i))
        train_order = np.random.permutation(train_X.shape[0])
        train_X = train_X[train_order]
        train_y = train_y[train_order]
        model.train_epoch(
            train_X,
            train_y,
            learning_rate=opt.learning_rate
        )
        valid_accs.append(model.evaluate(dev_X, dev_y))
        test_accs.append(model.evaluate(test_X, test_y))
        # Implemented by the group
        print('Valid acc:', valid_accs[-1], '| Test acc:', test_accs[-1]) 

    # plot
    plot(epochs, valid_accs, test_accs)
    


if __name__ == '__main__':
    main()
