<!--<h3><b>Colorful Image Colorization</b></h3>-->
# TDT05 Mini Project: Image Colorization
## Subject
The task deals with coloring in black and white images. The idea is to have self supervices machine learning models that can learn accurate values of color for black and white images and apply these to provided black and white images. The models should in essence predict the missing colors.

## Use of self-supervied learning

## Comparison with other models
Image in-coloring is a special task in that the other forms of learning, being supervised and unsupervised, are not really pplicable to the domain. This is because:
- Supervised learning requires labelling of data. The data from the image itself already contains the required labels, that being the values of the colors of the image. Labelling this manually would simply be double work
- For unsupervised learning, it is possible for a model to learn reresentations and recognize objects. It is even possible to associate colors with these representations. However, it is not possible to teach the model to _Correctly_ appy color without some data reference. If we train the model to just apply _some_ color to the objects it finds, we will be doing object detection as opposed to coloring.