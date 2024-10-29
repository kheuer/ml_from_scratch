"""This file contains the basic neuron and layer classes to be used to build with"""

import math
import random


class Neuron:
    """
    This class represnts an artificial neuron.
    It is a slave to its Layer.
    """

    def __init__(self, master):
        self.master = master
        self.bias = random.random()
        self.outgoing_edges = []
        self.incomming_edges = []


class Edge:
    """
    This is an Edge between two Neurons, it is a slave to its Layer.
    """

    def __init__(self, first_neuron, second_neuron):
        self.weight = random.random()
        self.first = first_neuron
        self.second = second_neuron

    def forward(self, val):
        activation_fn = self.second.master.activation_fn
        return activation_fn(val * self.weight + self.second.bias)


class FullyConnectedLayer:
    """
    This is a vanilla fully connected feedforward layer.
    It is a master to its Neurons and Edges.
    """

    def __init__(self, n_neurons, activation_fn):
        self.activation_fn = activation_fn
        self.neurons = [Neuron(self) for _ in range(n_neurons)]
        self.n_output_neurons = None

    def __repr__(self):
        return f"FullyConnectedLayer ({len(self.neurons)} --> {self.n_output_neurons})"

    def connect(self, next_layer):
        # create one connection for each pair of neurons
        for own_neuron in self.neurons:
            for foreign_neuron in next_layer.neurons:
                edge = Edge(own_neuron, foreign_neuron)

                own_neuron.outgoing_edges.append(edge)
                foreign_neuron.incomming_edges.append(edge)

        self.n_output_neurons = len(next_layer.neurons)

    def forward(self, vals):
        assert len(vals) == len(self.neurons)
        assert self.neurons
        assert self.n_output_neurons

        # initialize the output values
        output_vals = [0 for _ in range(self.n_output_neurons)]
        # for each input (prior layer or input data)
        for val in vals:
            # for each neuron in the layer
            for neuron in self.neurons:
                # for each connected neuron in the next layer
                for i, edge in enumerate(neuron.outgoing_edges):
                    output_vals[i] += edge.forward(val)
        return output_vals


class InputLayer(FullyConnectedLayer):
    """
    This is the input layer, it must be the first layer of every network and cannot be anywhere else
    """

    def __repr__(self):
        return f"InputLayer ({len(self.neurons)} --> {self.n_output_neurons})"


class OutputLayer(FullyConnectedLayer):
    """
    This is the output layer, it must be the last layer of every network and cannot be anywhere else
    """

    def __repr__(self):
        return f"OutputLayer ({len(self.neurons)} --> {self.n_output_neurons})"


class Model:
    def __init__(self, layers):
        assert len(layers) >= 2
        self.layers = layers
        for i, layer in enumerate(layers[:-1]):
            layer_next = layers[i + 1]
            layer.connect(layer_next)

    def forward(self, vals):
        for layer in self.layers[:-1]:
            vals = layer.forward(vals)
        return vals

    def __repr__(self):
        return f"Model with layers: {self.layers}"


def relu(val):
    """The Relu activation function"""
    return max(0, val)


def sigmoid(val):
    """The Sigmoid activation function"""
    return 1 / (1 + (math.e**-val))


def tanh(val):
    """The Tangent Hyperbolic activation function"""
    return math.tanh(val)


def leaky_relu(val):
    """The Leaky Relu activation function"""
    return max(0.1 * val, val)


def softmax(vals):
    """The Softmax activation function"""
    total_sum = sum([math.e**x_i for x_i in vals])
    return [math.e**x_i / total_sum for x_i in vals]
