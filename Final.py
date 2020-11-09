import sklearn
import numpy as np
import time
import h5py
from numpy.random import randn
from sklearn import metrics


set_M = h5py.File(r"C:\Users\samhi\OneDrive\Desktop\ML\Update\M_2710.mat")
set_F = h5py.File(r"C:\Users\samhi\OneDrive\Desktop\ML\Update\F_2813.mat")

dataset_M =  np.transpose(set_M['output'].value, axes=(1,0))
dataset_F =  np.transpose(set_F['output'].value, axes=(1,0))

dataset = np.r_[dataset_M, dataset_F]

labels_M = np.zeros((len(dataset_M),1), dtype=int) # set lable 0 for male
labels_F = np.ones((len(dataset_F),1), dtype=int) # set label 1 for female

data = dataset.reshape(len(dataset),27,1)
labels = np.r_[labels_M, labels_F]

print("Data Shape ", data.shape)
print("Labels Shape ", labels.shape)

len_x   = int(len(data))

# split into 90 : 10 train-test split

len_train = int(len_x*0.9)

len_test  = int(len_x*0.1)

print('len_train:', len_train)
print('len_test:', len_test)

idx = np.random.permutation(len_x)
frame_x,frame_y = data[idx], labels[idx]
print("Data Shape ", frame_x.shape)
print("Labels Shape ", frame_y.shape)

frame_x_train    = frame_x[0:len_train, :, :]
frame_y_train    = frame_y[0:len_train, ]
frame_x_test     = frame_x[len_train:len_x, :, :]
frame_y_test     = frame_y[len_train:len_x, ]


print("Train_Data Shape ", frame_x_train.shape)
print("Train_Labels Shape ", frame_y_train.shape)

print("Test_Data Shape ", frame_x_test.shape)
print("Test_Labels Shape ", frame_y_test.shape)

# defining softmax

def softmax( x):
    p = np.exp(x- np.max(x))
    return p / np.sum(p)

learn_rate = 0.01  
hidden_size = 64 #no.of nodes in hidden layer
input_size = 27 #no. of input features
output_size = 2 # either 0 or 1

#initialize weights and bias

Whh = randn(hidden_size, hidden_size) / 1000
Wxh = randn(hidden_size, input_size) / 1000
Why = randn(output_size, hidden_size) / 1000
bh = np.zeros((hidden_size, 1))
by = np.zeros((output_size, 1))
inputs = frame_x_train

#training - forward and backward pass

loss = 0
num_correct = 0
for i, x in enumerate(inputs):
    last_inputs = x
    h = np.zeros((Whh.shape[0], 1))
    last_hs = { 0: h }
    target = frame_y_train[i]
    #activation function - tanh
    h = np.tanh(Wxh @ x + Whh @ h + bh)
    last_hs[1] = h
    y = Why @ h + by
    probs = softmax(y)
    loss -= np.log(probs[target])
    num_correct += int(np.argmax(probs) == target)
    d_y = probs
    d_y[target] -= 1
    n = 1
    d_Why = d_y @ last_hs[n].T
    d_by = d_y
    d_Whh = np.zeros(Whh.shape)
    d_Wxh = np.zeros(Wxh.shape)
    d_bh = np.zeros(bh.shape)
    d_h = Why.T @ d_y
    for t in reversed(range(n)):
        temp = ((1 - last_hs[t + 1] ** 2) * d_h)
        d_bh += temp
        d_Whh += temp @ last_hs[t].T
        d_Wxh += temp @ last_inputs[:].T
        d_h = Whh @ temp
    for d in [d_Wxh, d_Whh, d_Why, d_bh, d_by]:
        np.clip(d, -1, 1, out=d)
    Whh -= learn_rate * d_Whh
    Wxh -= learn_rate * d_Wxh
    Why -= learn_rate * d_Why
    bh -= learn_rate * d_bh
    by -= learn_rate * d_by
    
print(loss/len_train)
print(num_correct/len_train)
