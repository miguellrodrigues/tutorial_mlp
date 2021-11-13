from math import sin, radians

from lib.network import load_network
import numpy as np

net = load_network('data/network.json')

x = 90

print(net.predict([x]), sin(radians(x)))
