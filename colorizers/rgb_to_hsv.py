import colorsys
import numpy as np


def array_to_image(array):
    r,g,b = array[0:1024].reshape((32,32)), array[1024:2048].reshape((32,32)), array[2048:3072].reshape((32,32))
    result_array = np.array([[r, g, b]], dtype=float)
    # normalize all elements in the array between 0 and 1
    result_array = result_array / 255
    return result_array

target = np.load("../datasets/cifar-100-python/train")

hsv_images = np.empty((len(target), 1, 3, 32, 32))

for image in range(len(target)):
    current = array_to_image(target[image])
    for i in range(32):
        for j in range(32):
            hsv_images[image][0][0][i][j], hsv_images[image][0][1][i][j], hsv_images[image][0][2][i][j] = colorsys.rgb_to_hsv(current[0][0][i][j], current[0][1][i][j], current[0][2][i][j])

print(hsv_images.shape)

for image in range(len(hsv_images)):
    for i in range(32):
        for j in range(32):
            hsv_images[image][0][0][i][j] *= 255
            hsv_images[image][0][1][i][j] *= 255
            hsv_images[image][0][2][i][j] *= 255

hsv_value = hsv_images.astype(np.int16)

np.save("../datasets/cifar-100-python/train_hsv", hsv_value)