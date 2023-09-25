# Image_colorisation
This project demonstrates the process of colorizing grayscale images using deep learning and the Caffe framework. The goal is to take black and white images as input and generate fully colorized versions of those images
 





# Project structure
* colorization_deploy_v2.prototxt: This file contains the architecture of the deep neural network used for colorization.


     Prototxt Link: 
     https://github.com/richzhang/coloriza...


* colorization_release_v2.caffemodel: This is the pre-trained model that is used for colorization. It contains the learned weights of the neural network.

     Model Link: 
     https://eecs.berkeley.edu/~rich.zhang...​

* pts_in_hull.npy: This NumPy file contains pre-defined color reference points used in the colorization process.

* black_white_to_color.py: This Python script demonstrates how to load the pre-trained model, process input images, and generate colorized output images.

# Getting started
1.Prerequisites:

* Python 3.x


* OpenCV (cv2)


* Caffe (withPython bindings)


* NumPy


2.Installation:

* Make sure you have all the required dependencies installed.


3.Usage:


* Run the black_white_to_color.py script to colorize a black and white image.
Ensure the input image is in grayscale format.


4.Output:

![image](https://github.com/vinitha09/Image_Colorisation/assets/88427641/d6c5e863-5eb7-48f9-b230-2be857f53451)
![image](https://github.com/vinitha09/Image_Colorisation/assets/88427641/8dd65ece-3eb5-4003-8cbb-c490279987c6)


