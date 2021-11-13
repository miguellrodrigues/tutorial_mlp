from lib.network import Network

network = Network([2, 8, 4, 1])

inputs = [
    [x, x] for x in range(12)
]

outputs = [
    [x/2] for x in range(12)
]


for _ in range(1000):
    for i in range(len(inputs)):
        network.train(inputs[i], outputs[i])

    error = network.global_error


for i in range(len(inputs)):
    predict = network.predict(inputs[i])

    print("Input = {} | Output = {} | Expected = {}".format(inputs[i], predict, outputs[i]))

