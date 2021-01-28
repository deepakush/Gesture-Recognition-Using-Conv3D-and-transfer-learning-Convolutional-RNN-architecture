## Gesture-Recognition-Using-Conv3D | Transfer-learning/CNN +RNN -architectures 
 
### Developed a real-time Hand Gesture Recognition system for cool feature in the Smart-TV 

### Problem Statement
As a data scientist at a home electronics company which manufactures state of the art smart televisions. We want to develop a cool feature in the smart-TV that can recognise five different gestures performed by the user which will help users control the TV without using a remote. 

![gesture_snap1](https://user-images.githubusercontent.com/40426356/87458252-562c7600-c627-11ea-8c24-e50cc51f5341.PNG)

### Understanding the Dataset
The training data consists of a few hundred videos categorized into one of the five classes. Each video (typically 2-3 seconds long) is divided into a sequence of 30 frames (images). These videos have been recorded by various people performing one of the five gestures in front of a webcam - similar to what the smart TV will use. 

![gesture_snap2](https://user-images.githubusercontent.com/40426356/87464407-b247c800-c630-11ea-937b-cd8386105236.PNG)

### Deep learning Architectures for analysing videos:

#### 1. 3D Convolutional Neural Networks (Conv3D)

3D convolutions are a natural extension to the 2D convolutions you are already familiar with. Just like in 2D conv, you move the filter in two directions (x and y), in 3D conv, you move the filter in three directions (x, y and z). In this case, the input to a 3D conv is a video (which is a sequence of 30 RGB images). If we assume that the shape of each image is 100 x 100 x 3, for example, the video becomes a 4D tensor of shape 100 x 100 x 3 x 30 which can be written as (100 x 100 x 30) x 3 where 3 is the number of channels. Hence, deriving the analogy from 2D convolutions where a 2D kernel/filter (a square filter) is represented as (f x f) x c where f is filter size and c is the number of channels, a 3D kernel/filter (a 'cubic' filter) is represented as (f x f x f) x c (here c = 3 since the input images have three channels). This cubic filter will now '3D-convolve' on each of the three channels of the (100 x 100 x 30) tensor

![gesture_snap3](https://user-images.githubusercontent.com/40426356/87458267-5cbaed80-c627-11ea-9fb4-637740eec0de.PNG)

#### 2. CNN + RNN architecture 

The conv2D network will extract a feature vector for each image, and a sequence of these feature vectors is then fed to an RNN-based network. The output of the RNN is a regular softmax (for a classification problem such as this one).

![gesture_snap4](https://user-images.githubusercontent.com/40426356/87458277-5f1d4780-c627-11ea-988c-48d160b81456.PNG)


### Data Pre-processing

•	**Resizing and cropping** of the images. This was mainly done to ensure that the NN only recognizes the gestures effectively rather than focusing on the other background noise present in the image.

•	**Normalization** of the images. Normalizing the RGB values of an image can at times be a simple and effective way to get rid of distortions caused by lights and shadows in an image.

•	At the later stages for improving the model’s accuracy, we have also made use of data augmentation, where we have slightly rotated the pre-processed images of the gestures in order to bring in more data for the model to train on and to make it more generalizable in nature as sometimes the positioning of the hand won’t necessarily be within the camera frame always.

![gesture_snap5](https://user-images.githubusercontent.com/40426356/87458294-63e1fb80-c627-11ea-981b-066f89cc185f.PNG)

. 

###             `Observation & Results for numerous tested NN architectures`

![gesture_snap6](https://user-images.githubusercontent.com/40426356/87458305-680e1900-c627-11ea-9cf4-4ab0b6e61f43.PNG)


`Reason:`

* **Best Acurracy**

* **Number of Parameters** less according to other models’ performance

*	**Learning rate** gradually decreasing after some Epochs



