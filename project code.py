import matplotlib.pyplot as plt
import matplotlib 
import numpy as np 
import pandas as pd 
import tensorflow as tf 
import tensorflow.keras.utils as tku
from sklearn.datasets import fetch_openml
mnist=fetch_openml("mnist_784")

X,y=mnist["data"],mnist["target"]
print(X.shape)
print(y.shape)

# visualizing the demo image
demo_img = X.iloc[1080].to_numpy()
demo_img=demo_img.reshape(28,28)# we reshape it into 28 x 28 pixels
plt.imshow(demo_img,cmap=matplotlib.cm.binary,interpolation="nearest")
plt.show()

#spliting the datasertraining the model
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

# convolutional neural networks
cnn=tf.keras.models.Sequential()
cnn.add(tf.keras.layers.Dense(units=64, activation="relu", input_dim=784))
cnn.add(tf.keras.layers.Dense(units=64, activation="relu"))
cnn.add(tf.keras.layers.Dense(units=10, activation="softmax"))# we use units=10 because we have 10 classes in MNIST dataset
# softmax is for categorical classes because we get probability for each carecter if the get prob for 1 is 0.1,prob for 2 is 0.001,prob for 9 is 0.97 ... the n thevalue 9 is the absolute value

#compiling the model
cnn.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])#we have categorical classes so we use categorical_crossentropy
cnn.fit(X_train,tku.to_categorical(y_train),epochs=25,batch_size=32)

#evaluating the model 
# in last code we train and evaluate the model in same line because the model donot know this is cat and this is dog it have to identify the features in the two classes and if we give the new image it identify this feature in the given image is similar to the 1st class so it gives high probability to the 1st class
#where as in this model we already know that this image is this we already know the values of the image so we want to evaluate the predicted value with the actual value of the image 
cnn.evaluate(X_test,tku.to_categorical(y_test))

#predicting the image
test_image=X.iloc[1080].to_numpy()
test_image=np.expand_dims(test_image,axis=0)
result=cnn.predict(test_image)
result=np.array(result)
prediction=np.argmax(result)
max_value_pred=np.max(result)
print("predicted Result:",prediction)
print("Probability of Predicted Result",max_value_pred)



