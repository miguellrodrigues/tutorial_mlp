import numpy as np

a1 = 25
a2 = 25


def ik(_x, _y):
    cos_theta2 = ((_x ** 2) + (_y ** 2) - (a1 ** 2) - (a2 ** 2)) / (2 * a1 * a2)
    theta_2 = np.arctan2(-np.sqrt(1 - (cos_theta2 ** 2)), cos_theta2)

    beta = np.arctan2(_y, _x)

    cos_phi = ((_x ** 2) + (_y ** 2) + (a1 ** 2) - (a2 ** 2)) / (2 * np.sqrt((_x ** 2) + (_y ** 2)) * a1)
    phi = np.arctan2(np.sqrt(1 - (cos_phi ** 2)), cos_phi)

    theta_1 = beta + phi

    return [theta_1 - (np.pi / 2.0), theta_2 - (np.pi / 2.0)]


inputs = []
iks = []

for i in range(1, 30):
  inputs.append([i, i])
  iks.append(ik(i, i)[1])
  
np.save('data/inputs.npy', inputs)
np.save('data/outputs.npy', iks)
