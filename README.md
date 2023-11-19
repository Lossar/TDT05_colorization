<!--<h3><b>Colorful Image Colorization</b></h3>-->
# TDT05 Mini Project: Image Colorization
## Subject
The task deals with coloring in black and white images. The idea is to have self supervised machine learning models that can learn accurate values of color for black and white images, and predict the missing colors when provided images without color.

Both a self-chosen architecture and other accomplished architectures from the field have been studied. The self-defined architecture is heavily inspired by the forked repo which is related to the code of a [paper](https://arxiv.org/pdf/1603.08511.pdf) on the subject of image coloring from 2016 by Richard Zhang, Phillip Isola, and Alexei A. Efros. Code for the self-defined architecture is present in this repository.

## Use of self-supervied learning
For this task, self-supevised learning is performed by taking input images and splitting them into three channels; hue, saturation, and value, and using the value as input while hue and saturation are the targets for the model. This provides the model with a black and white version of the image through only receiving the value, and the original colors of the image as its targets. A diagram of the training is provided below.

![Diagram of the training process](imgs/process.drawio.png)

This training is self-supervised in the way the data is its own label. For the training, there is nothing done to the data other than masking the target (hue and saturation) values by removing them from the input. There is no additional labelling performed as would be required for supervised learning, nor is the model completely without correcting factors as would be the case for unsupervised learning.
## Models
### Self-defined model
In order to get some new results and to train a model from start, an attempt was made to define a new model and train it on the CIFAR-100 dataset. CIFAR contains 50 000 images of dimension 32x32. The idea was that the smaller size would permit rapid training and evaluation as part of this project. However, the model did not achieve the results we were expecting, but the code can be found in this repo. Additionally, we showcase some results below from our experiments.

Prediction on image that was part of the training dataset
![Image of input, output, and original image](imgs_out/our_model/train_result.png?raw=true "Prediction on training data")

Prediction on an image that was not part of the dataset (unseen)
![Image of input, output, and original image](imgs_out/our_model/test_result.png?raw=true "Prediction on test data")

#### Running the self-defined model
To train and run the model it must be trained with prepared data. We have prepared datasets of images from CIFAR-100 with HSV colors and made them available for [download here](https://folk.ntnu.no/larsira/tdt05/).
To train and run from scratch:
- Download the `train_hsv` and `test_hsv` files and put them in the `datasets\cifar-100-python` folder
- Navigate to the `colorizers folder`
```
cd colorizers
```
- Go to train.py and set the hyper-parameters to your liking
- Select how many images you want to train on (up to 50000, which is the entire dataset). This is done by modifying the first slicer of the `samples` and `targets` variables. the below examples will give 100 samples and targets (0:100 as the first slicer). The amount of samples and targets *MUST BE EQUAL*, as they represent the same images but samples only contain the value channel while targets contain saturation and hue.
```
samples = samples[0:100, 2:3, 0:32, 0:32]
targets = targets[0:100, 0:2, 0:32, 0:32]
```
- When ready, run `train.py`. Losses will be printed to console
```
python train.py
```
- By default, the script will test the model after training by taking the first image from the loaded data, stripping the colors, and asking the model to predict the missing colors. To use a different image from the loaded data, change the `image_number` variable on line 64.
- Note also that by default, the script uses images from the `train_hsv` file. To use the test set, change the dataset_path variable string to
```
../datasets/cifar-100-python/test_hsv
```

The data have been converted from rgb to hsv using the `rgb_to_hsv.py` script present in the `colorizers` folder.

### Other self-supervised models
In order to present a realistic view of what is possible for image colorization, we also present and discuss some results of other models. Note that all training methods are self-supervised, but they vary in their implementation.

## Comparison with other models and ways of learning
Image in-coloring is a special task in that the other forms of learning, being supervised and unsupervised, are not applicable. This is because:
- Supervised learning requires labelling of data. The data from the image itself already contains the required labels, that being the values of the colors of the image. Labelling this manually would simply be double work
- For unsupervised learning, it is possible for a model to learn representations and recognize objects. It is even possible to associate colors with these representations. However, it is not possible to teach the model to _correctly_ apply color without some data reference. If we train the model to just apply _some_ color to the objects it finds, we will be doing object detection as opposed to coloring.

With the above being said, we can make comparisons between our model, older models, and today's state of the art. Although all perform self-supervised learning, they do so slightly different.

### Colorful Image Colorization

Colorful Image Colorization was regarded as one of the best black and white image colorization algorithms when it was released back in 2016. As mentioned in the subject introduction, this is what our project was heavily inspired by. Similarly to our algorithm, the training data is practically free because any color photo can be used as a training example by separating its channels into input and supervisory signal.

![Image of Colorful Image Colorization prediction](imgs_out/ECCV_SIGGRAPH_test.png?raw=true "Colorful Image Colorization Prediction")

The ECCV 16, which was released along with its research paper, creates relatively good colorized images, and according to them their methods successfully fooled humans on 32 % of the trials which beat any previous method by a significant margin, at the time of its release.

This model is the one this repo is forked from. This was done as experiments were started using the original code from that repo, and to properly reference back to the main source of inspiration for our own implementation.

### DeOldify: Self-Supervision through proprietary NoGAN
DeOldify is one of the biggest in-coloring models in terms of use. The model is available for use online through [DeepAI](https://deepai.org/machine-learning-model/colorizer), and as part of the image restoration tool offered by [MyHeritage](https://www.myheritage.no/incolor).

DeOldify is referred to by its author as being trained on "NoGAN". In traditional GAN, a generator and discriminator will be trained to either generate fake images (generator) or distinguish between real and fake images (discriminator). A key component to the NoGAN approach is that the generator is first trained in a self-supervised way only on the feature loss between its generations and expected outcomes. Only afterwards is traditional GAN training done. 

With regards to the GAN training itself, the discriminator is trained in such a way that it is automatically supervised. This is technically an internally built supervised problem, however it is worth noting that the supervision does not require human labelling. The labelling happens by the arcitechture itself, and while this does not qualify the learning as _self-supervised_ it still differs from traditional classes of supervised problems where humans normally create the labels.

Original|Colored
:------:|:-----:
![Original image](imgs/brown-Guernsey-cow-bw.png)|![Generated](imgs_out/deoldify/cow-colored.png)

The model does a good job of coloring in the images, with little bleeding between image components. The colored image is somewhat desaturated compared to the original. The generated image was aquired through the use of this [Google Collab](https://colab.research.google.com/github/jantic/DeOldify/blob/master/ImageColorizerColab.ipynb) found in the [DeOldify GitHub repository](https://github.com/jantic/DeOldify)

## Conclusion
Being a problem that lends itself well to self-superivision, coloring of images does not have any competetive implementations that are not self-supervised. The current self-supervised methods are however quite impressive especially when paired with ideas from not fully self-supervised methods like GANs. However, even in those scenarios, we can see from the DeOldify example that the self-supervised portion is still key to achieve great results.

Although we as a group did not manage to implement something very impressive, we still managed to apply self-supervised learning to the in-coloring problem. The results could be further improved by tuning the architecture with different layers.
