import json
import random
import sys
import mnist_loader
import network
import matplotlib.pyplot as plt
import numpy as np


def main(filename, num_node, num_epochs,
         training_cost_xmin=200, 
         test_accuracy_xmin=200, 
         test_cost_xmin=0, 
         training_accuracy_xmin=0,
         training_set_size=1000, 
         lmbda=0.0):
    """
    filename:           the name of the file where the results will be stored.  
    num_node:           the number of hidden node
    training_cost_xmin: the x asis min in training
    test_accuracy_xmin: the x asis min in test acc
    num_epochs:         the number of epochs to train for.
    training_set_size:  the number of images to train on.
    lmbda:              the regularization parameter. 
    """
    run_network(filename, num_node, num_epochs, training_set_size, lmbda)
    make_plots(filename, num_epochs, 
               test_accuracy_xmin,
               training_cost_xmin,
               test_accuracy_xmin, 
               training_accuracy_xmin,
               training_set_size)
                       
def run_network(filename, num_node, num_epochs, training_set_size=1000, lmbda=0.0):
    net = network.Network([784, num_node, 10], cost=network.CrossEntropyCost())
    net.large_weight_initializer()
    test_cost, test_accuracy, training_cost, training_accuracy \
        = net.SGD(training_data[:training_set_size], num_epochs, 10, 0.5,
                  evaluation_data=test_data, lmbda = lmbda,
                  monitor_evaluation_cost=True, 
                  monitor_evaluation_accuracy=True, 
                  monitor_training_cost=True, 
                  monitor_training_accuracy=True)
    f = open(filename, "w")
    json.dump([test_cost, test_accuracy, training_cost, training_accuracy], f)
    f.close()

def make_plots(filename, num_epochs, 
               training_cost_xmin=200, 
               test_accuracy_xmin=200, 
               test_cost_xmin=0, 
               training_accuracy_xmin=0,
               training_set_size=1000):
    f = open(filename, "r")
    test_cost, test_accuracy, training_cost, training_accuracy \
        = json.load(f)
    f.close()
    plot_training_cost(training_cost, num_epochs, training_cost_xmin)
    plot_test_accuracy(test_accuracy, num_epochs, test_accuracy_xmin)
    plot_test_cost(test_cost, num_epochs, test_cost_xmin)
    plot_training_accuracy(training_accuracy, num_epochs, 
                           training_accuracy_xmin, training_set_size)
    plot_overlay(test_accuracy, training_accuracy, num_epochs,
                 min(test_accuracy_xmin, training_accuracy_xmin),
                 training_set_size)

def plot_training_cost(training_cost, num_epochs, training_cost_xmin):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(training_cost_xmin, num_epochs), 
            training_cost[training_cost_xmin:num_epochs],
            color='#2A6EA6')
    ax.set_xlim([training_cost_xmin, num_epochs])
    ax.grid(True)
    ax.set_xlabel('Epoch')
    ax.set_title('Cost on the training data')
    plt.show()

def plot_test_accuracy(test_accuracy, num_epochs, test_accuracy_xmin):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(test_accuracy_xmin, num_epochs), 
            [accuracy/100.0 
             for accuracy in test_accuracy[test_accuracy_xmin:num_epochs]],
            color='#2A6EA6')
    ax.set_xlim([test_accuracy_xmin, num_epochs])
    ax.grid(True)
    ax.set_xlabel('Epoch')
    ax.set_ylim([50, 100])
    ax.set_title('Accuracy (%) on the test data')
    plt.show()

def plot_test_cost(test_cost, num_epochs, test_cost_xmin):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(test_cost_xmin, num_epochs), 
            test_cost[test_cost_xmin:num_epochs],
            color='#2A6EA6')
    ax.set_xlim([test_cost_xmin, num_epochs])
    ax.grid(True)
    ax.set_xlabel('Epoch')
    ax.set_title('Cost on the test data')
    plt.show()

def plot_training_accuracy(training_accuracy, num_epochs, 
                           training_accuracy_xmin, training_set_size):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(training_accuracy_xmin, num_epochs), 
            [accuracy*100.0/training_set_size 
             for accuracy in training_accuracy[training_accuracy_xmin:num_epochs]],
            color='#2A6EA6')
    ax.set_xlim([training_accuracy_xmin, num_epochs])
    ax.grid(True)
    ax.set_xlabel('Epoch')
    ax.set_ylim([50, 100])
    ax.set_title('Accuracy (%) on the training data')
    plt.show()

def plot_overlay(test_accuracy, training_accuracy, num_epochs, xmin,
                 training_set_size):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(xmin, num_epochs), 
            [accuracy/100.0 for accuracy in test_accuracy], 
            color='#2A6EA6',
            label="Accuracy on the test data")
    ax.plot(np.arange(xmin, num_epochs), 
            [accuracy*100.0/training_set_size 
             for accuracy in training_accuracy], 
            color='#FFA933',
            label="Accuracy on the training data")
    ax.grid(True)
    ax.set_xlim([xmin, num_epochs])
    ax.set_xlabel('Epoch')
    ax.set_ylim([50, 100])
    plt.legend(loc="lower right")
    plt.show()

def samplePlot(ind):
    pixels = np.array(training_data[ind][0]*255, dtype='uint8')
    pixels = pixels.reshape((28, 28))
    label = np.where(training_data[ind][1]==1)[0][0]
    plt.title('Label is {label}'.format(label=label))
    plt.imshow(pixels, cmap='gray')
    plt.show()

if __name__ == "__main__":
    filename = raw_input("Enter a file name: ")
    print 'loading dataset...'
    random.seed(12345678)
    np.random.seed(12345678)
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    det = int(raw_input("Want to make sample plot? 0:No, 1:yes"))
    if det == 1:
      ind = int(raw_input("Enter sample data to display (one number): "))
      samplePlot(ind)
    num_node = int(raw_input(
        "Enter the number of the neural node: "))
    num_epochs = int(raw_input(
        "Enter the number of epochs to run for: (suggest 200)"))
    training_cost_xmin = int(raw_input(
        "training_cost_xmin (suggest 100): "))
    test_accuracy_xmin = int(raw_input(
        "test_accuracy_xmin (suggest 100): "))
    test_cost_xmin = int(raw_input(
        "test_cost_xmin (suggest 0): "))
    training_accuracy_xmin = int(raw_input(
        "training_accuracy_xmin (suggest 0): "))
    training_set_size = int(raw_input(
        "Training set size (suggest 1000): "))
    lmbda = float(raw_input(
        "Enter the regularization parameter, lambda: "))
    main(filename, num_node, num_epochs, training_cost_xmin, 
         test_accuracy_xmin, test_cost_xmin, training_accuracy_xmin,
         training_set_size, lmbda)
