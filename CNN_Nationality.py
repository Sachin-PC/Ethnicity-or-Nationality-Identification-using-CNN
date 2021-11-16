import numpy as npy
import matplotlib.pyplot as matplot
% matplotlib inline
from keras.models import Sequential
from keras.layers.core import  Dense, Dropout,Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD,RMSprop, adam
from keras.utils import np_utils
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras import backend as K
K.set_image_dim_ordering('th')
import os
from PIL import Image
from numpy import *
import theano
seed = 1234
npy.random.seed(seed)
path1 = 'Image'
file_names = os.listdir(path1)
for i in range(len(file_names)):
    file_names[i] = file_names[i].split(".")[0]
    file_names[i] = file_names[i].split("_")[2]
file_names

images = os.listdir(path1)
path2 = 'Resized'
for image in images:
    im1 = Image.open(path1+'/'+image)
    img = im1.resize((200,200))
    grayimage =img.convert('L')
    grayimage.save(path2+'/'+image,"JPEG")

imagelist = os.listdir(path2)

imagematrix=array([array(Image.open('Resized'+'/'+ im2)).flatten()
              for im2 in imagelist],'f')
data,file_names = shuffle(imagematrix,file_names,random_state=2)
training_data=[data,file_names]

(X,y)=(training_data[0],training_data[1])
X_training,X_testing,y_training,y_testing=train_test_split(X,y,test_size=0.2,random_state=4)

X_training = X_training.reshape(X_training.shape[0],1,200,200)
X_testing = X_testing.reshape(X_testing.shape[0],1,200,200)
X_training = X_training.astype('float32')
X_testing = X_testing.astype('float32')

X_training/=255
X_testing/=255
y_training = npy.array(y_training,dtype='int64')
y_testing = npy.array(y_testing,dtype='int64')

for i in range(y_training.shape[0]):
    if(y_training[i] == 20170109142408075):
        y_training[i] = 1

        
for i in range(y_training.shape[0]):
    if(y_training[i] == 20170109150557335):
        y_training[i] = 1
        
for i in range(y_testing.shape[0]):
    if(y_testing[i] == 20170109150557335):
        y_testing[i] = 1

        
for i in range(y_testing.shape[0]):
    if(y_testing[i] == 20170109142408075):
        y_testing[i] = 1

y_training.shape
        
npy.unique(y_training)
npy.unique(y_testing)

y_testing = npy.array(y_testing)

y_training = np_utils.to_categorical(y_training)
y_testing = np_utils.to_categorical(y_testing)


num_classes = y_testing.shape[1]

CNNmodel = Sequential()
CNNmodel.add(Convolution2D(20,3,3,border_mode='valid',input_shape = (1,200,200)))
convout1 = Activation("relu")


CNNmodel.add(convout1)
CNNmodel.add(MaxPooling2D(pool_size = (2,2)))
CNNmodel.add(Convolution2D(10,(2,2),activation='relu'))
CNNmodel.add(MaxPooling2D(pool_size = (2,2)))
CNNmodel.add(Dropout(0.3))
CNNmodel.add(Flatten())
CNNmodel.add(Dense(128,activation='relu'))
CNNmodel.add(Dense(20,activation='relu'))
CNNmodel.add(Dense(num_classes,activation='softmax'))
CNNmodel.compile(optimizer = 'adam',metrics=['accuracy'],loss='categorical_crossentropy')

CNNmodel.fit(X_training,y_training,batch_size =15,epochs = 20,validation_data = (X_testing,y_testing),verbose=1)

# serialize CNNmodel to JSON
CNNmodel_json = CNNmodel.to_json()
with open("CNNmodel.json", "w") as json_file:
    json_file.write(CNNmodel_json)
# serialize weights to HDF5
CNNmodel.save_weights("CNNmodel.h5")
print("Saved CNNmodel to disk")
 
# later...
 
# load json and create CNNmodel
json_file = open('CNNmodel.json', 'r')
loaded_CNNmodel_json = json_file.read()
json_file.close()
loaded_CNNmodel = CNNmodel_from_json(loaded_CNNmodel_json)
# load weights into new CNNmodel
loaded_CNNmodel.load_weights("CNNmodel.h5")
print("Loaded CNNmodel from disk")
 
# evaluate loaded CNNmodel on test data
loaded_CNNmodel.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
score = loaded_CNNmodel.evaluate(X, Y, verbose=0)
print("%s: %.2f%%" % (loaded_CNNmodel.metrics_names[1], score[1]*100))
