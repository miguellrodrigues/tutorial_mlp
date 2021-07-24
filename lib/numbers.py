from random import SystemRandom
from datetime import datetime

sr = SystemRandom()
sr.seed(datetime.timestamp(datetime.now()))


def random_double(_min, _max):
    return sr.uniform(_min, _max)


def random_int(_min, _max):
    return sr.randint(_min, _max)
