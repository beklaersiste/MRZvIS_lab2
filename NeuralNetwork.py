import glob
import numpy
import random
from PIL import Image
from matplotlib import pyplot as plt


class NeuralNetwork:
    def __init__(self):
        self.weight = numpy.zeros((256, 256))

    def learn(self, path, use_numpy=False):
        files = glob.glob(path+'*')
        imgs_num = 20
        for file in range(imgs_num):
            vector = image_to_vector(files[file])
            self.weight = self.weight + (1/256)*(vector.T @ vector if use_numpy else mult_matrix(vector, transpose(vector)))
        for i in range(len(self.weight)):
            self.weight[i][i] = 0

    def recognize(self, noised_path, save_path, use_numpy=False):
        vector = image_to_vector(noised_path)
        result = vector.T
        vector = vector.T
        while True:
            result = self.weight @ vector if use_numpy else mult_matrix(self.weight, vector)
            sign(result)
            if numpy.array_equal(result, vector):
                vector_to_image(result.T, save_path)
                return
            else:
                vector = result

def transpose(matrix):
    result = numpy.zeros((len(matrix[0]), len(matrix)))
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            result[j][i] = matrix[i][j]
    return result

def mult_matrix(matrix1, matrix2):
    if len(matrix1[0]) == len(matrix2):
        result = numpy.zeros((len(matrix1), len(matrix2[0])))
        for i in range(len(matrix1)):
            for j in range(len(matrix2[0])):
                for k in range(len(matrix1[0])):
                    result[i][j] += matrix1[i][k]*matrix2[k][j]
        return result

def image_to_vector(path):
    array = numpy.asarray(Image.open(path).convert('RGB'))
    result = []
    for x in range(len(array)):
        for y in range(len(array[0])):
            result.append(-1.) if sum(array[x][y]) else result.append(1.)
    return numpy.array([result])

def vector_to_image(input, save_path):
    length = int(len(input[0])**0.5)
    result = [[[1., 1., 1.] if input[0][i*length+j] == -1 else [0., 0., 0.] for j in range(length)] for i in range(length)]
    plt.imsave(save_path, numpy.array(result))

def noise(path, save_path, percent=0.50):
    vector = image_to_vector(path)
    for _ in range(int(len(vector[0]) * percent)):
        i = random.randrange(len(vector[0]))
        vector[0][i] = -vector[0][i]
    vector_to_image(vector, save_path)

def sign(input):
    for i in range(len(input)):
        for j in range(len(input[0])):
            input[i][j] = -1 if input[i][j] < 0 else 1
