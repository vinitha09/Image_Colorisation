#importing packages
import numpy as np 
import cv2

print("loading models.....")


#loading a pre-trained deep neural network (DNN) model for image colorization. The model consists of two files: 'colorization_deploy_v2.prototxt' (the model architecture) 
#and 'colorization_release_v2.caffemodel' (the trained model weights)
net = cv2.dnn.readNetFromCaffe('colorization_deploy_v2.prototxt','colorization_release_v2.caffemodel')


#This line loads a NumPy array from the 'pts_in_hull.npy' file. These points represent color information for different color channels.y
pts = np.load('pts_in_hull.npy')

#You're getting the layer IDs for specific layers within the neural network model. These layers are used to fine-tune the colorization process.
class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")


#The 'pts' array is transposed and reshaped to match the expected shape for input to the neural network.
pts = pts.transpose().reshape(2,313,1,1)


# setting the blob values for the 'class8_ab' and 'conv8_313_rh' layers in the neural network. These blobs contain important data used during the colorization process.
net.getLayer(class8).blobs = [pts.astype("float32")]
net.getLayer(conv8).blobs = [np.full([1,313],2.606,dtype='float32')]

#input reading
image = cv2.imread('lion.jpg')


#The image is scaled by converting it to float32 and dividing by 255 to normalize pixel values between 0 and 1.
scaled = image.astype("float32")/255.0

#The image is converted from the BGR color space to the LAB color space. The LAB color space separates the image into a luminance channel (L) and two chrominance channels (A and B).
lab = cv2.cvtColor(scaled,cv2.COLOR_BGR2LAB)

#The LAB image is resized to dimensions (224, 224). This size is often used as input to neural networks.
resized = cv2.resize(lab,(224,224))

#The luminance channel (L) is extracted from the LAB image, and 50 is subtracted from all its pixel values.
L = cv2.split(resized)[0]
L -= 50

#he preprocessed L channel is set as input to the neural network using cv2.dnn.blobFromImage. This function prepares the data for the network.
net.setInput(cv2.dnn.blobFromImage(L))

#The network is used to predict the chrominance channels (A and B), which are obtained from the forward pass of the neural network and transposed into the proper order.
ab = net.forward()[0, :, :, :].transpose((1,2,0))

#The predicted chrominance channels are resized to match the dimensions of the original image.
ab = cv2.resize(ab, (image.shape[1],image.shape[0]))


#The original luminance channel (L) is combined with the predicted chrominance channels (A and B) to create a colorized LAB image.
L = cv2.split(lab)[0]
colorized = np.concatenate((L[:,:,np.newaxis], ab), axis=2)


#The colorized LAB image is converted back to the BGR color space using OpenCV.
colorized = cv2.cvtColor(colorized,cv2.COLOR_LAB2BGR)

colorized = np.clip(colorized,0,1)


#The pixel values are scaled back to the range [0, 255] and converted to unsigned 8-bit integers (uint8).
colorized = (255 * colorized).astype("uint8")


cv2.imshow("Original",image)
cv2.imshow("Colorized",colorized)

cv2.waitKey(0)
