from lib.network import Network

network = Network([2, 3, 1])

inputs = [
    [0, 0],
    [1, 1],
    [2, 2],
    [3, 3]
]

outputs = [
    [0],
    [2],
    [4],
    [6]
]


for _ in range(15000):
    for i in range(len(inputs)):
        network.train(inputs[i], outputs[i])

    error = network.global_error


for i in range(len(inputs)):
    predict = network.predict(inputs[i])

    print("Input = {} | Output = {} | Expected = {}".format(inputs[i], predict, outputs[i]))

print(network.predict([4, 8]))