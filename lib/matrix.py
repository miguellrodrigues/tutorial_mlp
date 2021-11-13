from lib.numbers import random_double
import numpy as np


def array_to_matrix(array):
    matrix = Matrix(len(array), 1)

    for i in range(len(array)):
        matrix.set_value(i, 0, array[i])

    return matrix


def matrix_to_array(matrix):
    array = np.zeros(matrix.rows)

    for i in range(matrix.rows):
        array[i] = matrix.get_value(i, 0)

    return array


class Matrix:
    def __init__(self, rows, cols, is_random=False):
        self.rows = rows
        self.cols = cols

        self.data = np.array([[.0] * cols for _i in range(rows)], dtype=np.float64)

        if is_random:
            for i in range(rows):
                for j in range(cols):
                    self.data[i][j] = random_double(0, 1)

    def set_value(self, row, column, value):
        self.data[row][column] = value

    def get_value(self, row, column):
        return self.data[row][column]

    def transpose(self):
        matrix = Matrix(self.cols, self.rows)

        matrix.data = self.data.transpose()

        return matrix

    def hadamard(self, mx):
        matrix = Matrix(mx.rows, mx.cols)

        matrix.data = self.data * mx.data

        return matrix

    def multiply(self, mx):
        matrix = Matrix(self.rows, mx.cols)

        matrix.data = self.data @ mx.data

        return matrix

    def add(self, mx):
        matrix = Matrix(self.rows, self.cols)

        matrix.data = self.data + mx.data

        return matrix

    def subtract(self, mx):
        matrix = Matrix(self.rows, self.cols)

        matrix.data = self.data - mx.data

        return matrix

    def scalar(self, x):
        matrix = Matrix(self.rows, self.cols)

        matrix.data = self.data * x

        return matrix

    def map(self, func):
        for i in range(self.rows):
            for j in range(self.cols):
                self.set_value(i, j, func(self.get_value(i, j)))

    def assign_array(self, array):
        count = 0

        for i in range(self.rows):
            for j in range(self.cols):
                self.set_value(i, j, array[count])
                count += 1
          
    def copy(self):
        matrix = Matrix(self.rows, self.cols)

        matrix.data = self.data.copy()

        return matrix

    def print(self):
        for i in range(self.rows):
            for j in range(self.cols):
                print(str(self.get_value(i, j)) + ' ', end='')

                if j == self.cols - 1:
                    print('')
