from lib.network import Network

rede = Network([2, 16, 16, 1])

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

for _ in range(10000):
    rede.train(inputs[i], [outputs[i]])

    i += 1

    if i == 4:
        i = 0

    error = rede.global_error

    print(error)


for i in range(len(inputs)):
    predict = rede.predict(inputs[i]).get_value(0, 0)

    print("Input = {} | Output = {} | Expected = {}".format(inputs[i], outputs[i], predict))
