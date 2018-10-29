# First need to import some libraries
import numpy as np
import tensorflow as tf
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import keras 
from keras.models import Sequential
from keras.layers import Dense
from keras.losses import binary_crossentropy
from keras.optimizers import Adam

# built-in that allows us to get a random distribution of 2 blobs
data = make_blobs(n_samples=50,n_features=2,centers=2,random_state=101)
data2 = make_blobs(n_samples=140,n_features=2,centers=2,random_state=101)

# data[0][0], data[0][1]
# store the x and y coordinates so that they can be plotted and visually seen
features = data[0]

# graph the two blobs
# %matplotlib inline
plt.scatter(features[:,0],features[:,1])
plt.show()

# get the class labels that I mentioned above
labels = data[1]

# plot the data in the 2 respective groups
plt.scatter(features[:,0],features[:,1], c=labels,cmap='coolwarm')
plt.show()

# Split the data into a group for training and a group for validating the training
xtrain, xtest, ytrain, ytest = train_test_split(features, labels, random_state=101)
# print(len(xtrain), len(xtest))

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

model.save("{}.keras".format("First-keras-ANN"))

# Get validation data
xtrain2, _, _, _ = train_test_split(data2[0][-30:],data2[1][-30:])
predictions = model.predict(xtrain2)

# List comprehension for determining the labels
predictions = np.array([round(i[0]) for i in predictions.tolist()])

# Make sure X, Y and Labels are all of same shape; so no shape mismatch will occur
a,b,c = xtrain2[:,0], xtrain2[:,1], predictions
assert (a.shape == b.shape == c.shape)
print("X, Y and labels have passed assertion step.")

plt.scatter(a,b,c=c)
plt.scatter(features[:,0],features[:,1], c=labels,cmap='coolwarm')
plt.show()

# Future implementation in straight tensorflow // Lower level API
# https://github.com/soerendip/Tensorflow-binary-classification/blob/master/Tensorflow-binary-classification-model.ipynb
