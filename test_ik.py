# from lib.network import Network
# import numpy as np
# import matplotlib.pyplot as plt
#
#
# network = Network([2, 5, 1])
#
# inputs = np.load('data/inputs.npy')
# outputs = np.load('data/outputs.npy')
#
# min_o = np.min(outputs)
# max_o = np.max(outputs)
#
#
# def normalize(x, _mi, _ma):
#     return (x - _mi) / (_ma - _mi)
#
#
# def denormalize(x, _mi, _ma):
#     return x * (_ma - _mi) + _mi
#
#
# outputs = normalize(outputs, min_o, max_o)
#
# in_max = np.max(inputs)
# in_min = np.min(inputs)
#
# errors = []
#
# for _ in range(1000):
#     for i in range(len(inputs)):
#         network.train(inputs[i], [outputs[i]])
#
#     error = network.global_error
#     errors.append(error)
#     print(error)
#
#
# predicts = []
#
# network.save('data/network.json')
#
# for i in range(len(inputs)):
#     predicts.append(network.predict(inputs[i]))
#
# for i in range(len(inputs)):
#     print("Input = {} | Output = {} | Expected = {}".format(inputs[i], predicts[i], outputs[i]))
#
# predicts = denormalize(np.array(predicts), min_o, max_o)
# outputs = denormalize(outputs, min_o, max_o)
#
# fig, axs = plt.subplots(1, 2, tight_layout=True)
#
# axs[0].plot(inputs, outputs)
# axs[0].plot(inputs, predicts, 'o')
#
# axs[1].plot(errors)
# axs[1].legend(['Error'])
#
# plt.show()

from lib.network import Network

network = Network([1, 5, 1])

inputs = [
    [x] for x in range(10)
]

outputs = [
    [x * 2] for x in range(10)
]

for _ in range(100):
    for i in range(len(inputs)):
        network.train(inputs[i], outputs[i])

for i in range(len(inputs)):
    print("Expected {} | Predicted {}".format(outputs[i], network.predict(inputs[i])))