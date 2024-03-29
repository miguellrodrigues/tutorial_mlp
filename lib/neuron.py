import numpy as np


def sigmoid_activation(x):
    return x


def sigmoid_derived(x):
    return x if x > 0 else 0


class Neuron:
    def __init__(self, value):
        self.value = value

        self.activated_value = .0
        self.derived_value = .0

    def set_value(self, value):
        self.value = value

        self.activate()
        self.derive()

    def activate(self):
        self.activated_value = sigmoid_activation(self.value)

    def derive(self):
        self.derived_value = sigmoid_derived(self.activated_value)

    def get_value(self):
        return self.value

    def get_activated_value(self):
        return self.activated_value

    def get_derived_value(self):
        return self.derived_value
