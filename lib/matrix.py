from lib.numbers import random_double


def array_to_matrix(array):
    matrix = Matrix(len(array), 1)

    for i in range(len(array)):
        matrix.set_value(i, 0, array[i])

    return matrix


class Matrix:
    def __init__(self, rows, cols, is_random=False):
        self.rows = rows
        self.cols = cols

        self.data = [[.0] * cols for _i in range(rows)]

        if is_random:
            for i in range(rows):
                for j in range(cols):
                    self.data[i][j] = random_double(-.001, .001)

    def set_value(self, row, column, value):
        self.data[row][column] = value

    def get_value(self, row, column):
        return self.data[row][column]

    def transpose(self):
        matrix = Matrix(self.cols, self.rows)

        for i in range(self.rows):
            for j in range(self.cols):
                matrix.set_value(j, i, self.get_value(i, j))

        return matrix

    def hadamard(self, mx):
        matrix = Matrix(mx.rows, mx.cols)

        for i in range(mx.rows):
            for j in range(mx.cols):
                matrix.set_value(i, j, self.get_value(i, j) * mx.get_value(i, j))

        return matrix

    def multiply(self, mx):
        matrix = Matrix(self.rows, mx.cols)

        for i in range(matrix.rows):
            for j in range(matrix.cols):
                aux = .0

                for k in range(mx.rows):
                    aux += self.get_value(i, k) * mx.get_value(k, j)

                matrix.set_value(i, j, aux)

        return matrix

    def add(self, mx):
        matrix = Matrix(self.rows, self.cols)

        for i in range(self.rows):
            for j in range(self.cols):
                matrix.set_value(i, i, self.get_value(i, j) + mx.get_value(i, j))

        return matrix

    def subtract(self, mx):
        matrix = Matrix(self.rows, self.cols)

        for i in range(self.rows):
            for j in range(self.cols):
                matrix.set_value(i, i, self.get_value(i, j) - mx.get_value(i, j))

        return matrix

    def scalar(self, x):
        matrix = Matrix(self.rows, self.cols)

        for i in range(self.rows):
            for j in range(self.cols):
                matrix.set_value(i, i, self.get_value(i, j) * x)

        return matrix

    def assign_array(self, array):
        count = 0

        for i in range(self.rows):
            for j in range(self.cols):
                self.set_value(i, j, array[count])
                count += 1

    def print(self):
        for i in range(self.rows):
            for j in range(self.cols):
                print(str(self.get_value(i, j)) + ' ', end='')

                if j == self.cols - 1:
                    print('')
