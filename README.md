<!--<h3><b>Colorful Image Colorization</b></h3>-->
# TDT05 Mini Project: Image Colorization
## Subject
The task deals with coloring in black and white images. The idea is to have self supervices machine learning models that can learn accurate values of color for black and white images and apply these to provided black and white images. The models should in essence predict the missing colors.

Both a self-chosen arcitecthure and other accomplished arcitecthures from the field have been studied. The self-defined arcitecthure is heavily inspired by the forked repo which is related the code of a [paper](https://arxiv.org/pdf/1603.08511.pdf) on the subject of image coloring from 2016 by Richard Zhang, Phillip Isola, and Alexei A. Efros. Code for the self-defined arcitecthure is present in this repository.

## Use of self-supervied learning
For this task, self-supevised learning is performed by taking input images, splitting the images into three channels; Hue, Value, and saturation, and using the value as input while Hue and Saturation are the targets for the model. 

This training is self-supervised by the way the data is its own label. For the training, there is nothing done to the data other than masking the target values by removing them from the input. There is no additional labelling performed as would be required for supervised learning, nor is the model completely without correcting factors as would be the case for unsupervised learining.
## Models
### Self-defined model
In order to get some new results and to train a model from start, an attempt was made to define a new model and train it on the CIFAR-100 dataset. CIFAR contains many tiny images of dimension 32x32. The idea was that the smaller size would permit rapid training and evaluation as part of this project. The model did not achieve the results we were expecting, however the code can be found in this repo. Additionally, we showcase some results below from our experiments.

### Other self-supervised models
In order to present a realistic view of what is possible for image colorization, we also present some results of other models, and a brief overview of their arcitechtures and ways of training. Note that all training methods are self-supervised, but they vary in the implementation.

## Comparison with other models and ways of learning
Image in-coloring is a special task in that the other forms of learning, being supervised and unsupervised, are not really pplicable to the domain. This is because:
- Supervised learning requires labelling of data. The data from the image itself already contains the required labels, that being the values of the colors of the image. Labelling this manually would simply be double work
- For unsupervised learning, it is possible for a model to learn reresentations and recognize objects. It is even possible to associate colors with these representations. However, it is not possible to teach the model to _Correctly_ appy color without some data reference. If we train the model to just apply _some_ color to the objects it finds, we will be doing object detection as opposed to coloring.

With the above being said, we can make comparisons between our model, older models, and the state of the art. Although all perform self-supervised learning, they do it slightly differently.