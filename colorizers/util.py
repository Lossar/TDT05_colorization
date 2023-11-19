from PIL import Image, ImageEnhance
import numpy as np
from skimage import color
import torch
import torch.nn.functional as F
from IPython import embed

# Make image B&W
def grayscale(np_image):
    # Convert to grayscale
    color_channel_size = int(len(np_image) / 3)

    for i in range(0, color_channel_size):
        color = 0.07 * np_image[i] + 0.72 * np_image[i + color_channel_size] + 0.21 * np_image[i + color_channel_size * 2]
        np_image[i], np_image[i + color_channel_size], np_image[i + color_channel_size * 2] = color, color, color

    return np_image


# Load training dataset
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
    return dict

def grayConversion(image):
    grayValue = 0.07 * image[:,:,2] + 0.72 * image[:,:,1] + 0.21 * image[:,:,0]
    gray_img = grayValue.astype(np.uint8)
    return gray_img

# Input: b&w
# Train/validation: color and b&w