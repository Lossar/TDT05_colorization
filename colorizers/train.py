import tarfile
import numpy as np
import torch
from torch import optim, nn
from torch.utils.data import DataLoader
import cv2

from dataset import Dataset
from colorizer_model import ColorizerModel

import trainer

# Number of epochs in training
epochs = 50

# Learning rate
learning_rate = 0.01

# Load samples and targets (I redefined the cifar-100 dataset - removed everything but the images themselves)
samples = np.load("../datasets/cifar-100-python/train_hsv_samples")
targets = np.load("../datasets/cifar-100-python/train_hsv_targets")

samples = samples[0:100, 0:1, 2:3, 0:32, 0:32]
targets = targets[0:100, 0:1, 0:2, 0:32, 0:32]

# Create tensors from data arrays
input_samples = torch.from_numpy(samples).float()
input_targets = torch.from_numpy(targets).float()

# Put samples and targets into a dataset
dataset = Dataset(input_samples, input_targets)
train_dataloader = DataLoader(dataset, batch_size=4000, shuffle=True)

# Create a model from network
model = ColorizerModel() #TODO: define/create a neural network

# Create a trainer
trainer = trainer.Trainer(
    model=model,
    dataset=dataset,
    loss_fn=nn.MSELoss(),
    optimizer=optim.Adam(model.parameters(), lr=learning_rate)  #TODO: choose optimizer (I just chose one of the more common ones for the time being)
)

# Train the model
trainer.train(epochs=epochs)

# Save the trained model
torch.save(model.state_dict(), "trained_model.pt")


# Check if prediction is working
'''data = np.load("../datasets/cifar-100-python/test_grayscale")

model = Network()
model.load_state_dict(torch.load("trained_model.pt"))
input_data = torch.from_numpy(data[0])
input_data = input_data.to(torch.float32)
prediction = model(input_data).cpu()
image = prediction.detach().numpy()

import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
#image = data[0]
#image = util.grayscale(image)
image = image.reshape(3,32,32).transpose(1,2,0)
plt.imshow(image)
plt.show()'''