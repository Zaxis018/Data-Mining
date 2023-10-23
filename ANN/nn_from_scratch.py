import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import math 

global w1,b1,w2,b2

random_matrix = np.random.randn(10,5)

def relu(x):
    temp = x.flatten()
    non_negative_values = []
    for i in range(len(temp)):
        if temp[i] > 0:
            non_negative_values.append(temp[i])
        else:
            non_negative_values.append(0)
    return np.array(non_negative_values).reshape(x.shape)

plt.scatter(random_matrix,relu(random_matrix))

def derivative_relu(x):
  temp = x.flatten()
  non_negative_values = []
  for i in range(len(temp)):
    if temp[i] > 0:
        non_negative_values.append(1)
    else:
        non_negative_values.append(0)
  return np.array(non_negative_values).reshape(x.shape)

plt.scatter(random_matrix,derivative_relu(random_matrix))

def softmax(x):
    x_exp = np.exp(x)
    # Calculate the sum of the exponentiated values for normalization.
    sum_exp = np.sum(x_exp)
    # Calculate the softmax probabilities.
    softmax_probs = x_exp / sum_exp
    return softmax_probs

plt.scatter(random_matrix,softmax(random_matrix))

def initialize_weights():
  w1=np.random.randn(784,10)
  b1=np.random.randn(10)
  w2=np.random.randn(10,10)
  b2=np.random.randn(10)
  return w1,b1,w2,b2

w1,b1,w2,b2=initialize_weights()

import tensorflow as tf
mnist = tf.keras.datasets.mnist

# Split the data into training and testing sets
(images, labels), _ = mnist.load_data()
images = images / 255.0

# Flatten the 28x28 images into a 1D array (784 elements)
images_flat =images.reshape(images.shape[0], -1)

# Create a dictionary to store the data
data_dict = {
    "label": labels,
}

# Add flattened pixel values to the dictionary
for i in range(images_flat.shape[1]):
    data_dict[f"pixel_{i}"] = images_flat[:, i]

# Create the DataFrame
df = pd.DataFrame(data_dict)
label_df=pd.DataFrame(df.iloc[:, 0])
images_df=pd.DataFrame(df.drop(columns=['label']))
#dataframe to matrix
image_matrix=images_df.values
label_column=label_df.values
image_matrix[1].shape

random_index = np.random.randint(0, len(images_df))
random_index
plt.imshow(image_matrix[random_index].reshape(28,28))

#train and test split
x_train=image_matrix[:50000]
x_test =image_matrix[50000:]
y_train=label_column[:50000]
y_test=label_column[50000:]
print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)

def forward_pass(data):
  X=data
  z1=X@w1+b1
  a1=relu(z1)
  z2=a1@w2+b2
  a2=softmax(z2)
  return z1,a1,z2,a2

z1,a1,z2,a2= forward_pass(x_train)

z1.shape,a1.shape,z2.shape,a2.shape

image_matrix.shape

y_train.shape

w2.shape

"""##Loss and backpropagation equation derivation

"""

def update_weights(learning_rate,x_batch,y_batch,one_hot_encoded,w1,b1,w2,b2):
  z1, a1, z2, a2 = forward_pass(x_batch)
  Y = y_batch
  Y=one_hot_encoded
  dz2=a2-Y#...eqn1
  dw2=1/32*dz2.T@a1 #...eqn2

  db2=1/32*sum(dz2)

  dz1=dz2@w2*derivative_relu(z1)#...eqn3    * =elementwise matrix multiplication
  dw1=1/32*(dz1.T@x_batch)#...eqn4
  db1=1/32*(sum(dz1))#...eqn5
  print('w1',w1)
  w1=w1-learning_rate*(dw1.T)
  print('w1',w1)
  b1=b2-learning_rate*db1
  w2=w2-learning_rate*dw2
  b2=b2-learning_rate*db2
  return w1,b1,w2,b2

one_hot_encoded1 = np.zeros((32,10))
one_hot_encoded1[np.arange(32),y_train[2:34].flatten()] = 1

e,f,g,h=update_weights(0.1,x_train[2:34],y_train[2:34],one_hot_encoded1,w1,b1,w2,b2)

def calculate_accuracy(dataset, labels):
    # Perform forward pass to get a, b, c, and d
    a, b, c, d = forward_pass(dataset)
    predicted_labels = d  #predicted labels
    # Calculate accuracy
    correct_predictions = (predicted_labels == labels).sum()
    total_predictions = len(labels)
    accuracy = correct_predictions / total_predictions
    return accuracy

y_train[2:34].shape

def train_nn(train_data, test_data, epochs, lr, batch_size):
    x_train1=train_data
    y_train1=test_data

    num_batches = len(train_data) // batch_size
    w1, b1, w2, b2 =initialize_weights()
    for i in range(epochs):
        for batch_start in range(0, len(train_data), batch_size):
            batch_end = batch_start + batch_size
            batch_x = x_train1[batch_start:batch_end]
            batch_y = y_train1[batch_start:batch_end]
            # Apply one-hot encoding for the current batch
            num_samples = batch_y.shape[0]
            num_classes = 10
            one_hot_encoded = np.zeros((num_samples, num_classes))
            one_hot_encoded[np.arange(num_samples), batch_y.flatten()] = 1

            w1, b1, w2, b2 = update_weights(lr, batch_x, batch_y,one_hot_encoded,w1,b1,w2,b2)

        print('Epoch:', i+1)
        print('Training accuracy:', calculate_accuracy(x_train, y_train))
        print('Validation accuracy:', calculate_accuracy(x_test, y_test))

train_nn(x_train, y_train, epochs=1, lr=0.5, batch_size=32)


