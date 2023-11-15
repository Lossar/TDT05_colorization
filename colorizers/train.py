import colorsys
import tarfile
import numpy as np
import torch
from torch import optim, nn
from torch.utils.data import DataLoader
import cv2

import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

from dataset import Dataset
from colorizer_model import ColorizerModel

import trainer

# Number of epochs in training
epochs = 1000

# Learning rate
learning_rate = 0.25

# Load samples and targets (I redefined the cifar-100 dataset - removed everything but the images themselves)
samples = np.load("../datasets/cifar-100-python/train_hsv")
targets = np.load("../datasets/cifar-100-python/train_hsv")

samples = samples[0:1, 2:3, 0:32, 0:32]
targets = targets[0:1, 0:2, 0:32, 0:32]

# Create tensors from data arrays
input_samples = torch.from_numpy(samples).float()
input_targets = torch.from_numpy(targets).float()
# Put samples and targets into a dataset
dataset = Dataset(input_samples, input_targets)
train_dataloader = DataLoader(dataset, batch_size=4000, shuffle=True)

# Create a model from network
model = ColorizerModel().cuda()

# Create a trainer
trainer = trainer.Trainer(
    model=model,
    dataset=dataset,
    loss_fn=nn.MSELoss(),
    optimizer=optim.Adadelta(model.parameters(), lr=learning_rate)
)

# Train the model
trainer.train(epochs=epochs)

# Save the trained model
torch.save(model.state_dict(), "trained_model.pt")


# Check if prediction is working
data = np.load("../datasets/cifar-100-python/train_hsv")

image_number = 0

model = ColorizerModel()
model.load_state_dict(torch.load("trained_model.pt"))
image_original = data[image_number]
input_data = torch.from_numpy(image_original[2:3, 0:32, 0:32])
input_data = input_data.to(torch.float32)
prediction = model(input_data).cpu()
image_color = prediction.detach().numpy()

image_value = input_samples[image_number]
image = np.zeros((32, 32, 3), dtype=float)

counter = 0
for i in range(32):
    for j in range(32):
        image[i][j][2] = image_value.flatten()[counter]
        image[i][j][0] = image_color[0].flatten()[counter]
        image[i][j][1] = image_color[1].flatten()[counter]
        counter += 1

fig, (ax1, ax2, ax3) = plt.subplots(figsize=(15,15), ncols=3, nrows=1)
ax1.set_title("Input image")
ax1.imshow(np.moveaxis(np.array(image_value), 0, 2) / 255, cmap='gray')

ax2.set_title("Prediction")
ax2.imshow(mpl.colors.hsv_to_rgb(image.reshape((32, 32, 3)) / 255))

ax3.set_title("Actual")
ax3.imshow(mpl.colors.hsv_to_rgb(np.moveaxis(data[image_number],0,2)/255))

plt.show()