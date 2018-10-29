# google tensorflow to predict classes in sklearnmake_blobs
# First need to import some libraries
import numpy as np
import tensorflow as tf
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# built-in that allows us to get a random distribution of 2 blobs
data = make_blobs(n_samples=50,n_features=2,centers=2,random_state=101)
data2 = make_blobs(n_samples=140,n_features=2,centers=2,random_state=101)
# in jupyer: play with this tho, should show the x and y coordinates along with the label (which group)
# data[0][0], data[0][1]

# store the x and y coordinates so that they can be plotted and visually seen
features = data[0]

# graph the stuff
# might have to that inline command since jupyter/ check it out
# %matplotlib inline
plt.scatter(features[:,0],features[:,1])

# get the class labels that I mentioned above
labels = data[1]

# plot the data in the 2 respective groups
# check if c in plt.scatter is classes; explain cmap is just the way you want to style the data points
plt.scatter(features[:,0],features[:,1], c=labels,cmap='coolwarm')
#plt.show()

# now we get into the tensorflow portion
# start by splitting the data into a group for training and a group for validating the training
xtrain, xtest, ytrain, ytest = train_test_split(features, labels, random_state=101)
# print(len(xtrain))
# print(len(xtest))
# input()
# create an input function, essentially how the model takes in the input training data

################
# CHECK THIS LINE B
################
# input_function = tf.estimator.inputs.numpy_input_fn(x=xtrain, y=ytrain, num_epochs=1000,shuffle=True)

# # Create the model
# ##################
# # Check the right way to use the estimator thing in tensorlflow
# #################
# # model = tf.estimator(n_classes=2) # tf.estimator.LinearClassifier()

# model.train(input_fn=input_function,steps=100)

# # then gotta do the prediction step


import keras 
# gotta figure out this one tho
# import tensorflow.keras as keras
from keras.models import Sequential
from keras.layers import Dense
from keras.losses import binary_crossentropy
from keras.optimizers import Adam

NUM_EPOCHS = 10
model = Sequential()
model.add(Dense(10,input_shape=xtrain.shape[1:],activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

# check this out to make sure the stuff is right
model.compile(loss=binary_crossentropy,
			optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999),
			metrics=['accuracy'])
model.fit(xtrain,ytrain,epochs=NUM_EPOCHS,batch_size=1,validation_data=(xtest,ytest))

model.save("{}.keras".fomrat("First-keras-ANN"))

# Something along these lines for prediction in keras
predictions = []
#print(xtrain.shape, xtrain.shape[1:],xtrain[0])
xtrain2, _, _, _ = train_test_split(data2[0][-30:],data2[1][-30:])
# print("NOW:")
#print(model.predict(xtrain,batch_size=1))
# for i in xtrain2:
# 	i[0] += 0.5
predictions = model.predict(xtrain2)
# print("PRED",predictions)
# print("LABELS", data2[1], data2[1].shape)
predictions = np.array([round(i[0]) for i in predictions.tolist()])
# print(predictions.shape)
# print(xtrain2)
# print(data2)

print(xtrain2[:,0], xtrain2[:,0].shape)
print(data2[0][:,0], data2[0][:,0].shape)
print(predictions, predictions.shape)
a,b,c = xtrain2[:,0], xtrain2[:,1], predictions
assert (a.shape == b.shape == c.shape)
print("passed the assertion step")
plt.show()
plt.scatter(a,b,c=c)
plt.scatter(features[:,0],features[:,1], c=labels,cmap='coolwarm')
#plt.scatter(data2[0][:,0],data2[0][:,1],c=predictions,cmap='seismic')
plt.show()




# https://github.com/soerendip/Tensorflow-binary-classification/blob/master/Tensorflow-binary-classification-model.ipynb