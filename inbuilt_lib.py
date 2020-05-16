#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sklearn
import numpy as np
from scipy import io
import tensorflow as tf
import os
import time
import h5py
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import math
import pandas as pd
import glob, os


# In[ ]:


set_M = h5py.File('/Users/nxs162330/Documents/Clean_Speech/M_2710.mat')
set_F = h5py.File('/Users/nxs162330/Documents/Clean_Speech/F_2813.mat')


# In[ ]:


dataset_M =  np.transpose(set_M['output'].value, axes=(1,0))
dataset_F =  np.transpose(set_F['output'].value, axes=(1,0))

dataset = np.r_[dataset_M, dataset_F]


# In[ ]:


labels_M = np.zeros((len(dataset_M),1), dtype=int)
labels_F = np.ones((len(dataset_F),1), dtype=int)


# In[ ]:


data = dataset.reshape(len(dataset),27,1)
labels = np.r_[labels_M, labels_F]

print("Data Shape ", data.shape)
print("Labels Shape ", labels.shape)


# In[ ]:


len_x   = int(len(data))
len_train = int(len_x*0.8)

len_test  = int(len_x*0.2)

print('len_train:', len_train)
print('len_test:', len_test)


# In[ ]:


idx = np.random.permutation(len_x)
frame_x,frame_y = data[idx], labels[idx]
print("Data Shape ", frame_x.shape)
print("Labels Shape ", frame_y.shape)


# In[ ]:


frame_x_train    = frame_x[0:len_train, :, :]
frame_y_train    = frame_y[0:len_train, ]
frame_x_test     = frame_x[len_train:len_x, :, :]
frame_y_test     = frame_y[len_train:len_x, ]


print("Train_Data Shape ", frame_x_train.shape)
print("Train_Labels Shape ", frame_y_train.shape)

print("Test_Data Shape ", frame_x_test.shape)
print("Test_Labels Shape ", frame_y_test.shape)


# In[ ]:


def one_hot_encode(idx, vals=2):
    '''
    For use to one-hot encode the 10- possible labels
    '''
    len_idx=len(idx)
    out = np.zeros((len_idx, vals))
    for i in range(len_idx):
        out[i,idx[i]] = 1
        
    return out


# In[ ]:


frame_y_test


# In[ ]:


frame_y_train = one_hot_encode(frame_y_train,2)
frame_y_test  = one_hot_encode(frame_y_test,2)


# In[ ]:


class DataHelper():
    
    def __init__(self):
        self.i = 0
        self.k = 0
        
        #self.all_train_batches = [data_batch1,data_batch2,data_batch3,data_batch4,data_batch5]
        #self.test_batch = [test_batch]
        
        self.training_data = None
        self.training_labels = None
        
        self.test_data = None
        self.test_labels = None
    
    def set_up_data(self,two_ch_input_training,y_training,two_ch_input_test,y_test):
        
        print("Setting Up Training Data and Labels")
        
        self.training_data= frame_x_train 
        train_len = len(self.training_data)
        
        self.training_labels = frame_y_train
        
        print("Setting Up Test Images and Labels")
        
        self.test_data = frame_x_test
        test_len = len(self.test_data)
        
        self.test_labels = frame_y_test

        
    def next_batch(self, batch_size):
        x = self.training_data[self.i:self.i+batch_size]
        y = self.training_labels[self.i:self.i+batch_size]
        self.i = (self.i + batch_size) % len(self.training_data)
        return x, y
    
    def next_batch_test(self, batch_size):
        x_test = self.test_data[self.k:self.k+batch_size]
        y_test = self.test_labels[self.k:self.k+batch_size]
        #print(self.k)
        self.k = (self.k + batch_size) % len(self.test_data)
        
        return x_test, y_test


# In[ ]:


dh = DataHelper()
dh.set_up_data(frame_x_train,frame_y_train,frame_x_test,frame_y_test)


# In[ ]:


learning_rate = tf.placeholder(tf.float32)
keep_prob = tf.placeholder(tf.float32,name = "keep_prob")

# with tf.name_scope("inputs"):
x = tf.placeholder(tf.float32,[None,27,1],"x-input")
y_true = tf.placeholder(tf.float32,[None,2], "y-true")  


# In[ ]:


num_layers = 4
num_hidden = 100
nDataSamples = len(data)
batch_size = 500


# In[ ]:


def rnn_cell():
        return tf.contrib.rnn.BasicRNNCell(num_units=num_hidden, activation=tf.nn.relu)
    

stacked_rnn = tf.contrib.rnn.MultiRNNCell([rnn_cell() for _ in range(num_layers)])

final_outputs, final_state = tf.nn.dynamic_rnn(cell=stacked_rnn,
                                   inputs=x,
                                   dtype=tf.float32)


# In[ ]:


final_outputs.shape


# In[ ]:


flat = tf.reshape(final_outputs,[-1,27*100])


# In[ ]:


def init_weights(shape):
    init_random_dist=tf.truncated_normal(shape,stddev=0.05)
    return tf.Variable(init_random_dist)

def init_bias(shape):
    init_bias_vals=tf.ones(shape)/10
    return tf.Variable(init_bias_vals)

def normal_full_layer(input_layer,size):
    input_size=int(input_layer.get_shape()[1])
    W=init_weights([input_size,size])
    b=init_bias([size])
    return tf.matmul(input_layer,W)+b

full_layer_one = tf.nn.relu(normal_full_layer(flat,512))
Y4 = tf.nn.dropout(full_layer_one, keep_prob)

full_layer_one.shape


# In[ ]:


W5 = tf.Variable(tf.truncated_normal([512, 2], stddev=0.05))
B5 = tf.Variable(tf.ones([2])/10)
Y  = tf.nn.softmax(tf.matmul(Y4, W5) + B5)
Y = tf.identity(Y, name="output_t")

# with tf.name_scope("loss"):
cross_entropy = -tf.reduce_sum(y_true*tf.log(tf.clip_by_value(Y,1e-10,1.0)))
is_correct = tf.equal(tf.argmax(Y,1), tf.argmax(y_true,1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

# with tf.name_scope("train"):
optimizer = tf.train.AdamOptimizer(learning_rate)
train_op  = optimizer.minimize(cross_entropy) 


# In[ ]:


batch_size=100
size_test= int(len_test/batch_size)

print(size_test)

size_train= int(len_train/batch_size)
print(size_train)


# In[ ]:


checkpoint_dir = "/Users/nxs162330/Documents/Clean_Speech/full_model/"
tf.gfile.MakeDirs(checkpoint_dir)

learningRates = np.hstack((1e-3*np.ones(10),
                           1e-4*np.ones(6),
                           1e-5*np.ones(4)))
nEpochs = len(learningRates)


with tf.Session() as sess:
    saver = tf.train.Saver()
    tf.train.write_graph(sess.graph_def,
                         checkpoint_dir,
                         "graph.pbtxt",
                         True)
    sess.run(tf.global_variables_initializer())    
    
    for epoch in np.arange(nEpochs):
        train_batch_count = 0 
        train_batch_acc_total = 0
        for i in range(size_train):
            batch = dh.next_batch(batch_size)
            sess.run(train_op, feed_dict={x: batch[0], y_true: batch[1],learning_rate: learningRates[epoch],keep_prob: 0.7})# hold_prob: 0.5 
            
            
            if i%1000 == 0:
                print('Currently on training step {}'.format(i))


            # PRINT OUT A MESSAGE EVERY 100 STEPS
            if i%(size_train-1) == 0 and i!=0:
                print('Currently on step {} for Testing'.format(i))
                
                test_batch_acc_total = 0
                test_batch_count = 0
                

                for k in range (size_test):
                    batch_test = dh.next_batch_test(batch_size)
                    test_batch_acc_total += sess.run(accuracy,feed_dict={x:batch_test[0],y_true:batch_test[1],learning_rate: learningRates[epoch],keep_prob: 1.0}) #,hold_prob:1.0
                    test_batch_count += 1
                    train_batch_acc_total += sess.run(accuracy,feed_dict={x:batch[0],y_true:batch[1],learning_rate: learningRates[epoch],keep_prob: 1.0}) #,hold_prob:1.0
                    train_batch_count += 1


                    if k%1000 == 0:
                        print('Currently on testing step {}'.format(k))

                print('Epoch: {}'.format(epoch))
                print('Testing Accuracy: {}\n'.format(test_batch_acc_total/test_batch_count))
                print('Training Accuracy: {}\n'.format(train_batch_acc_total/train_batch_count))
                
                
        tf.gfile.MakeDirs(checkpoint_dir + '/model' + str(epoch))
        checkpoint_file = os.path.join(checkpoint_dir + '/model' + str(epoch), "model")
        saver.save(sess, checkpoint_file)
        print("**** SAVED MODEL ****")      
        print("**** COMPLETED EPOCH ****")


# In[ ]:


batch[1].shape


# In[ ]:


batch[0].shape


# In[ ]:


learningRates[epoch]


# In[ ]:


i


# In[ ]:


frame_y_test


# In[ ]:




