# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 02:06:48 2018

@author: abhayakumar
"""

###################################################### To check version of anaconda
#Go to annaconda command prompt
#conda info


###################################################### To install tensorflow
#1.Go to anaconda command prompt(right click --> run as administrator)
#2.conda install -c conda-forge tensorflow
#3.Press y when prompted

#--------------OR----------------------

#1.Go to tools above
#2.open command prompt
#3.conda install tensorflow
#4.Press y when prompted



############### Dealing with variables usually
import numpy as np
###
import tensorflow as tf

###############
data = np.random.randint(1000, size=10000)

x = tf.constant(data, name='x')
y = tf.Variable(5*(x**2) -3*x +15 , name='y')
###
print(y)
###
model=tf.global_variables_initializer()
###
with tf.Session() as session:
    session.run(model)
    print(session.run(y))
###
###############
############### Visualization of the graph on tensorboard
 
with tf.Session() as session:
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("H:\C\Project 01\Training_Session 01\Python_Training_Session 01\Concise_Training 01", session.graph)
    
################To view the graph, python -m tensorboard.main --logdir=/path/to/logs

################ Using Placeholders 
#import tensorflow as tf

df1 = pd.DataFrame(np.random.normal(loc=1,size=(10000,1)).reshape((100,100)))
df2 = pd.DataFrame(np.random.normal(loc=5,size=(10000,1)).reshape((100,100)))
x = tf.placeholder("float", None)
y = x**2 + 10

with tf.Session() as session:
    row_mean=[]
    for i in range(len(df1)):
        result = session.run(y,feed_dict={x: df1.iloc[i]})
        row_mean.append(np.mean(result))
    print(row_mean)
    print(np.mean(row_mean))
   # writer = tf.summary.FileWriter("/tmp/basic", session.graph)
with tf.Session() as session:
    row_mean=[]
    for i in range(len(df2)):
        result = session.run(y,feed_dict={x: df2.iloc[i]})
        row_mean.append(np.mean(result))
    print(row_mean)
    print(np.mean(row_mean))   
################
###################################### Broadcasting
a = tf.constant([[1, 2, 3], [4, 5, 6]], name='a')
b = tf.constant([[1, 2, 3], [4, 5, 6]], name='b')
add_op = a + b

with tf.Session() as session:
    print(session.run(add_op))
############
a = tf.constant([[1, 2, 3], [4, 5, 6]], name='a')
b = tf.constant(100, name='b')
add_op = a + b

with tf.Session() as session:
    print(session.run(add_op))
############
a = tf.constant([[1, 2, 3], [4, 5, 6]], name='a')
b = tf.constant([100, 101, 102], name='b')
add_op = a + b

with tf.Session() as session:
    print(session.run(add_op))
#############
a = tf.constant([[1, 2, 3], [4, 5, 6]], name='a')
b = tf.constant([[100], [101]], name='b')
add_op = a + b

with tf.Session() as session:
    print(session.run(add_op))
    
################################################################################
######################################################################################## keras
###################################################### To install keras
#1.Go to anaconda command prompt(right click --> run as administrator)
#2.conda install -c conda-forge keras
#3.Press y when prompted

#--------------OR----------------------

#1.Go to tools above
#2.open command prompt
#3.conda install keras
#4.Press y when prompted

##################################################################
###################################################################
#Larger CNN for the MNIST Dataset
%reset -f
import PIL
import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load data
(X_train1, y_train), (X_test, y_test) = mnist.load_data()
PIL.Image.fromarray(X_train1[0])    #to see/visualise the training images

# reshape to be [samples][pixels][width][height]
X_train = X_train1.reshape(X_train1.shape[0], 1, 28, 28).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')
# normalize inputs from 0-255 to 0-1
X_train = X_train / 255.0
X_test = X_test / 255.0
# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
# define the larger model


# create model
model = Sequential()
model.add(Conv2D(30, (5, 5), input_shape=(1, 28, 28), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(15, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='softmax'))  #10 is the total number of classes in data (0 to 9)
# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
#adam 97.97%

# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=2, batch_size=256, shuffle=True)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Deep CNN Accuracy: ", scores[1]*100, "%") 


model.predict(X_test[0:1])   
