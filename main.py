from lib.network import Network
from lib.numbers import random_int

network = Network([2, 16, 1])

inputs = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]

outputs = [
    0,
    1,
    1,
    1
]

error = 10
i = 0

for _ in range(4000):
    network.train(inputs[i], [outputs[i]])

    i = random_int(0, 3)

    error = network.global_error


for i in range(len(inputs)):
    predict = network.predict(inputs[i]).get_value(0, 0)

    print("Input = {} | Output = {} | Expected = {}".format(inputs[i], predict, outputs[i]))
