import matplotlib.pyplot as plt
import plotly
import numpy as npy
from sklearn.svm import SVC
import plotly.plotly as py
import plotly.graph_objs as go
from plotly import tools
from sklearn.model_selection import GridSearchCV
import pandas as pd  
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.utils import shuffle
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
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
path2 = 'Resize'
for image in images:
    im1 = Image.open(path1+'/'+image)
    img = im1.resize((200,200))
    grayimage =img.convert('L')
    grayimage.save(path2+'/'+image,"JPEG")

imagelist = os.listdir(path2)
imagematrix=array([array(Image.open('Resize'+'/'+ im2)).flatten()
              for im2 in imagelist],'f')
data,file_names = shuffle(imagematrix,file_names,random_state=2)
training_data=[data,file_names]

(X,y)=(training_data[0],training_data[1])
X_training,X_testing,y_training,y_testing=train_test_split(X,y,test_size=0.2,random_state=4)
X_training = X_training.reshape(X_training.shape[0],1,200,200)
X_testing = X_testing.reshape(X_testing.shape[0],1,200,200)
X_training = X_training.astype('float32')
X_testing = X_testing.astype('float32')X_training/=255
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

print(y_training.shape)
print(X_training.shape)        
npy.unique(y_training)
npy.unique(y_testing)
y_testing = npy.array(y_testing)
# Compute a PCA 
n_components = 100
X_training = X_training.reshape(8545,40000)
#data.reshape((data.shape[0], data.shape[1], 1))
print(X_training.shape) 
pca = PCA(n_components=n_components, whiten=True).fit(X_training)

X_testing = X_testing.reshape(2137,40000)
print(X_testing.shape)
X_train_pca = pca.transform(X_training)
X_test_pca = pca.transform(X_testing)
print("Fitting the classifier to the training set")
#t0 = time()
param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
clf = clf.fit(X_train_pca, y_training)
#print("done in %0.3fs" % (time() - t0))
print("Best estimator found by grid search:")
print(clf.best_estimator_)
#print("Predicting people's names on the test set")
#t0 = time()
n_classes=5
target_names =[" White", "Black", "Asian", "Indian","Others"]
y_pred = clf.predict(X_test_pca)
#print("done in %0.3fs" % (time() - t0))

print(classification_report(y_testing, y_pred,target_names=target_names))
print(confusion_matrix(y_testing, y_pred,labels=range(n_classes)))
