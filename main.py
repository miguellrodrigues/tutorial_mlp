from lib.network import Network

network = Network([2, 8, 16, 1])

inputs = [
    [x, x] for x in range(4)
]

outputs = [
    2*x for x in range(4)
]


for _ in range(100):
    for i in range(len(inputs)):
        network.train(inputs[i], [outputs[i]])

    error = network.global_error


for i in range(len(inputs)):
    predict = network.predict(inputs[i]).get_value(0, 0)

    print("Input = {} | Output = {} | Expected = {}".format(inputs[i], predict, outputs[i]))
