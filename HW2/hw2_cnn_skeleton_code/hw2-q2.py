#!/usr/bin/env python

# Deep Learning Homework 2

import argparse

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torchvision
from matplotlib import pyplot as plt
import numpy as np

import utils


# Configure Device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Device:', device)

class CNN(nn.Module):
    
    def __init__(self, dropout_prob):
        """
        The __init__ should be used to declare what kind of layers and other
        parameters the module has. For example, a CNN module has convolution,
        max pooling, activation, linear, and other types of layers. For an 
        idea of how to us pytorch for this have a look at
        https://pytorch.org/docs/stable/nn.html
        """
        super(CNN, self).__init__()
        
        ########## Define our CNN which is comprised by multiple layers #########
        
        # (1) First convolutional block
        # We had to define the first convolutional block this way to plot the activation maps
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=5, stride=1, padding=2)
        self.activ_func1 =  nn.ReLU()
        self.max_pooling1 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        # (2) Second convolutional block
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2)           
        )
        
        # (3) First fully connected block
        self.fc1 = nn.Sequential( 
            nn.Flatten(), #Transforming an 6x6 matrix into a 1D array
            nn.Linear(in_features = 16*6*6, out_features = 600), # First affine transformation
            nn.ReLU(), # Activation function
        )
        
        # (3)
        # (I) Dropout with p dropout probability
        self.dropout = nn.Dropout(p = 0.3) 
            
        # (4) Second fully connected block
        self.fc2 = nn.Sequential(
            nn.Linear(in_features = 600, out_features = 120), # Second affine transformation 
            nn.ReLU() # Activation function
        )
        
        # (6) Output layer
        self.output = nn.Sequential(         
            nn.Linear(in_features = 120, out_features = 10), # n_classes = 10
            nn.LogSoftmax(dim = 1) # Output layer with softmax
        )
        
        
    def forward(self, x):
        """
        x (batch_size x n_channels x height x width): a batch of training 
        examples

        Every subclass of nn.Module needs to have a forward() method. forward()
        describes how the module computes the forward pass. This method needs 
        to perform all the computation needed to compute the output from x. 
        This will include using various hidden layers, pointwise nonlinear 
        functions, and dropout. Don't forget to use logsoftmax function before 
        the return

        One nice thing about pytorch is that you only need to define the
        forward pass -- this is enough for it to figure out how to do the
        backward pass.
        """
        
        # (1) Reshape the input image into the correct size
        # CNN's only receive images (batch_size, n_channels, H, W)
        size = x.shape[0]
        x = torch.reshape(x,(size,1,x.shape[1]))
        x = torch.reshape(x,(size,1,28,28)) 
        
        # (2) Foward pass
        
        # (I) First convolutional block
        output = self.conv1(x)
        output = self.activ_func1(output)
        output = self.max_pooling1(output)
        
        # (II) Second convolutional block
        output = self.convblock2(output)
        
        # (III) First fully connected block
        output = self.fc1(output)
        
        # (IV) Only apply dropout during the training
        if self.training:
            output = self.dropout(output)
            
        # (V) Second fully connected block
        output = self.fc2(output)
        
        # (VI) Output block
        output = self.output(output)

        return output

def train_batch(X, y, model, optimizer, criterion, **kwargs):
    """
    X (n_examples x n_features)
    y (n_examples): gold labels
    model: a PyTorch defined model
    optimizer: optimizer used in gradient step
    criterion: loss function

    To train a batch, the model needs to predict outputs for X, compute the
    loss between these predictions and the "gold" labels y using the criterion,
    and compute the gradient of the loss with respect to the model parameters.

    Check out https://pytorch.org/docs/stable/optim.html for examples of how
    to use an optimizer object to update the parameters.

    This function should return the loss (tip: call loss.item()) to get the
    loss as a numerical value that is not part of the computation graph.
    """
    # (1) Model training
    model.train()
    
    # (2) Clear gradients
    optimizer.zero_grad()
    
    # (3) Forward step
    y_hat = model(X)
    loss = criterion(y_hat, y)

    # (4) Backward step
    loss.backward()
    
    # (5) Optimize step
    optimizer.step()
    
    return loss.item()


def predict(model, X):
    """X (n_examples x n_features)"""
    scores = model(X)  # (n_examples x n_classes)
    predicted_labels = scores.argmax(dim=-1)  # (n_examples)
    return predicted_labels


def evaluate(model, X, y):
    """
    X (n_examples x n_features)
    y (n_examples): gold labels
    """
    model.eval()
    y_hat = predict(model, X)
    n_correct = (y == y_hat).sum().item()
    n_possible = float(y.shape[0])
    model.train()
    return n_correct / n_possible


def plot(epochs, plottable, ylabel='', name=''):
    plt.clf()
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.plot(epochs, plottable)
    plt.show()
    #plt.savefig('%s.pdf' % (name), bbox_inches='tight')


activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

def plot_feature_maps(model, train_dataset):
    
    model.conv1.register_forward_hook(get_activation('conv1'))
    
    data, _ = train_dataset[4]
    data.unsqueeze_(0)
    output = model(data)

    plt.imshow(data.reshape(28,-1)) 
    plt.show()
    #plt.savefig('original_image.pdf')
    
    plt.clf()

    k=0
    act = activation['conv1'].squeeze()
    fig,ax = plt.subplots(2,4,figsize=(12, 8))
    
    for i in range(act.size(0)//3):
        for j in range(act.size(0)//2):
            ax[i,j].imshow(act[k].detach().cpu().numpy())
            k+=1 
            #plt.savefig('activation_maps.pdf') 
            
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-epochs', default=20, type=int,
                        help="""Number of epochs to train for. You should not
                        need to change this value for your plots.""")
    parser.add_argument('-batch_size', default=8, type=int,
                        help="Size of training batch.")
    parser.add_argument('-learning_rate', type=float, default=0.0005,
                        help="""Learning rate for parameter updates""")
    parser.add_argument('-l2_decay', type=float, default=0)
    parser.add_argument('-dropout', type=float, default=0.8)
    parser.add_argument('-optimizer', choices=['sgd', 'adam'], default='adam')
    
    
    opt = parser.parse_args()

    utils.configure_seed(seed=42)

    data = utils.load_classification_data()
    dataset = utils.ClassificationDataset(data)
    train_dataloader = DataLoader(
        dataset, batch_size=opt.batch_size, shuffle=True)
    dev_X, dev_y = dataset.dev_X, dataset.dev_y
    test_X, test_y = dataset.test_X, dataset.test_y

    # initialize the model
    model = CNN(opt.dropout)
    
    # get an optimizer
    optims = {"adam": torch.optim.Adam, "sgd": torch.optim.SGD}

    optim_cls = optims[opt.optimizer]
    optimizer = optim_cls(
        model.parameters(), lr=opt.learning_rate, weight_decay=opt.l2_decay
    )
    
    # get a loss criterion
    criterion = nn.NLLLoss()
    
    print("-------------- Model --------------")
    print("epochs:",opt.epochs)
    print("Learning_rate:",opt.learning_rate)
    print("dropout:",opt.dropout)
    print("batch_size:",opt.batch_size)
    print("optimizer:",opt.optimizer)
    print("-----------------------------------")
    
    # training loop
    epochs = np.arange(1, opt.epochs + 1)
    train_mean_losses = []
    valid_accs = []
    train_losses = []
    for ii in epochs:
        print('Training epoch {}'.format(ii))
        for X_batch, y_batch in train_dataloader:
            loss = train_batch(
                X_batch, y_batch, model, optimizer, criterion)
            train_losses.append(loss)

        mean_loss = torch.tensor(train_losses).mean().item()
        print('Training loss: %.4f' % (mean_loss))

        train_mean_losses.append(mean_loss)
        valid_accs.append(evaluate(model, dev_X, dev_y))
        print('Valid acc: %.4f' % (valid_accs[-1]))

    print('Final Test acc: %.4f' % (evaluate(model, test_X, test_y)))
    
    # plot
    config = "{}-{}-{}-{}".format(opt.learning_rate, opt.dropout, opt.l2_decay, opt.optimizer)

    plot(epochs, train_mean_losses, ylabel='Loss', name='CNN-training-loss-{}'.format(config))
    plot(epochs, valid_accs, ylabel='Accuracy', name='CNN-validation-accuracy-{}'.format(config))
    
    plot_feature_maps(model, dataset)

if __name__ == '__main__':
    main()
