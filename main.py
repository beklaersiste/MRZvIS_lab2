from NeuralNetwork import NeuralNetwork, noise
from configs import *
import random
import glob

if __name__ == '__main__':
    nn = NeuralNetwork()
    nn.learn(pngs_path, True)
    test_img_path = random.choice(glob.glob(pngs_path+'*'))
    noise(test_img_path, noise_path, 0.2)
    nn.recognize(noise_path, recognize_path, True)




