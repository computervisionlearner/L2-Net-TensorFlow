#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 21:58:19 2017

@author: fs
"""

import tensorflow as tf
import cv2
import numpy as np
rate = 0.01
IMAGE_WIDTH = 32
IMAGE_HEIGHT = 32
q = 2.

def _conv(name, inputs, size, input_channels, output_channels, reuse = False):
  """ 　纯卷积   """
  with tf.variable_scope(name, reuse = reuse):

    kernel = _weight_variable('weights', shape=[size, size ,input_channels, output_channels])
    biases = _bias_variable('biases',[output_channels])
    out = tf.nn.bias_add(_conv2d(inputs, kernel),biases)

  return out

def _conv2d(value, weight):
  """conv2d returns a 2d convolution layer with full stride."""
  return tf.nn.conv2d(value, weight, strides=[1, 1, 1, 1], padding='SAME')

def _conv_s2(name, inputs, size, input_channels, output_channels, reuse = False):
  """ 　纯卷积   """
  with tf.variable_scope(name, reuse = reuse):

    kernel = _weight_variable('weights', shape=[size, size ,input_channels, output_channels])
    biases = _bias_variable('biases',[output_channels])
    out = tf.nn.bias_add(_conv2d_s2(inputs, kernel),biases)

  return out

def _conv2d_s2(value, weight):
  """conv2d returns a 2d convolution layer with full stride."""
  return tf.nn.conv2d(value, weight, strides=[1, 2, 2, 1], padding='SAME')

def _max_pool_2x2(value, name):
  """max_pool_2x2 downsamples a feature map by 2X."""
  with tf.variable_scope(name):
    return tf.nn.max_pool(value, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME', name=name)


def _weight_variable(name, shape, mean=0):
  """weight_variable generates a weight variable of a given shape."""
  initializer = tf.truncated_normal_initializer(mean=mean,stddev=0.1)
  var = tf.get_variable(name,shape,initializer=initializer, dtype=tf.float32)
  return var


def _bias_variable(name, shape):
  """bias_variable generates a bias variable of a given shape."""
  initializer = tf.constant_initializer(0.1)
  var = tf.get_variable(name, shape, initializer=initializer,dtype=tf.float32)
  return var

def _batch_norm(name, inputs, reuse = False):
  """ batch Normalization
  """
  #equals to tf.nn.batch_normalization() just when batch=1
  with tf.variable_scope(name, reuse = reuse):
    scale = 1   #shape=(64,)
    offset = 0   #shape=(64,)
    mean, variance = tf.nn.moments(inputs, axes=[0,1,2], keep_dims=True)  # (1, 1, 1, 64) nomornize in axis=0&1
    epsilon = 1e-5
    inv = tf.rsqrt(variance + epsilon) #rsqrt(x)=1./sqrt(x)
    normalized = (inputs-mean)*inv
    return scale*normalized + offset

def global_CNN(name, inputs, reuse = False):
  """ 　全卷积   """
  with tf.variable_scope(name, reuse = reuse):
    
    size1, size2, input_channels = inputs.shape.as_list()[1:]
    kernel = _weight_variable('weights', shape=[size1, size2 ,input_channels, input_channels])
    biases = _bias_variable('biases',[input_channels])
    out = tf.nn.conv2d(inputs, kernel, strides=[1, 1, 1, 1], padding='VALID')
    out = tf.nn.bias_add(out, biases)

  return out
    
    
    

def get_features(images, reuse=False):
  images = tf.image.resize_bilinear(images,(32,32))
  conv1 = _conv('conv1', images, 3, 1, 32, reuse = reuse) #(batch,32,32,32)
  conv1 = _batch_norm('norm1',conv1,reuse = reuse)
  conv1 = tf.nn.relu(conv1, name = 'relu1')
  assert conv1.shape.as_list()[1:]==[32,32,32]

  conv2 = _conv('conv2', conv1, 3,32,32,reuse = reuse)#(batch,32,32,32)
  conv2 = _batch_norm('norm2',conv2,reuse = reuse)
  conv2 = tf.nn.relu(conv2, name = 'relu2')
  assert conv2.shape.as_list()[1:]==[32,32,32]

  conv3 = _conv_s2('conv3', conv2, 3,32,64,reuse = reuse)#(batch,16,16,64)
  conv3 = _batch_norm('norm3',conv3,reuse = reuse)
  conv3 = tf.nn.relu(conv3, name = 'relu3')
  assert conv3.shape.as_list()[1:]==[16,16,64]
  
  conv4 = _conv('conv4', conv3, 3,64,64,reuse = reuse)#(batch,16,16,64)
  conv4 = _batch_norm('norm4',conv4,reuse = reuse)
  conv4 = tf.nn.relu(conv4, name = 'relu4')
  assert conv4.shape.as_list()[1:]==[16,16,64]
  
  
  conv5 = _conv_s2('conv5', conv4, 3,64,128,reuse = reuse)#(batch,8,8,128)
  conv5 = _batch_norm('norm5',conv5,reuse = reuse)
  conv5 = tf.nn.relu(conv5, name = 'relu5')
  assert conv5.shape.as_list()[1:]==[8,8,128]
  
  conv6 = _conv('conv6', conv5, 3,128,128,reuse = reuse)#(batch,8,8,128)
  conv6 = _batch_norm('norm6',conv6,reuse = reuse)
  conv6 = tf.nn.relu(conv6, name = 'relu6')
  assert conv6.shape.as_list()[1:]==[8,8,128]

  conv7 = global_CNN('conv7', conv6, reuse = reuse)#(batch,1,1,128)
  conv7 = _batch_norm('norm7',conv7,reuse = reuse)

  conv7 = tf.squeeze(conv7, name="squeeze")
  
  out = tf.nn.l2_normalize(conv7,dim=-1,name="lrn")

  return conv7, out

def get_distance(f1, f2):
  """ 
  
  features1 / features2  p*q  
  (p = the number of the 3Dpoints in a batch) 
  
  """
  result = tf.matmul(f1, f2, transpose_b=True)   
  D = tf.sqrt(2.*tf.subtract(1.,result))
  
  return D
  
def loss1(matrix):
  """ 
  输入一个矩阵,矩阵的每个元素是两两特征向量的欧式距离　
  返回第一个损失函数   
  """
  similarity = tf.subtract(2., matrix)
  s_c = tf.nn.softmax(similarity, dim=-1,name="s_c")
  s_r = tf.nn.softmax(similarity, dim=0,name="r_c")
  
  E1 = -0.5*(tf.reduce_mean(tf.log(tf.diag_part(s_c))) + \
                      tf.reduce_mean(tf.log(tf.diag_part(s_r))))
  
  tf.summary.scalar('E1', E1)
  return E1
  
def loss2(f1,f2):
  """
  输入的特征向量为　　最后一层BN的输出，　即在LRN之前
  此函数中的f1/f2与get_distance函数中的l1/l2不同
  """
  R1 = tf.divide(tf.matmul(f1,f1,transpose_a=True),128)
  R2 = tf.divide(tf.matmul(f2,f2,transpose_a=True),128)
  
  diag1 = tf.diag_part(R1)
  diag2 = tf.diag_part(R2)
  
  E2 = (0.5*(tf.reduce_sum(tf.square(R1))+tf.reduce_sum(tf.square(R2))-\
       tf.reduce_sum(tf.square(diag1))-tf.reduce_sum(tf.square(diag2))))/128.
  
  tf.summary.scalar('E2', E2)
  return E2
  

def loss3(map1,map2):
  """
  输入的特征图  可能是(batch,1,1,C)形式的
  也可能是(batch,h,w,c)形式的
  暂时只讨论第一种形式的
  """
    
    
  G = tf.matmul(map1, map2, transpose_b = True)
    
  v_c = tf.nn.softmax(G, dim=-1,name="v_c")
  v_r = tf.nn.softmax(G, dim=0, name="v_r")
    
  E3=-0.5*(tf.reduce_sum(tf.log(tf.diag_part(v_c))) +\
                tf.reduce_sum(tf.log(tf.diag_part(v_r))))
  
  tf.summary.scalar('E3', E3)
    
  return E3
  


def caculate_loss(map1, map2, out1, out2):
  with tf.variable_scope('caculate_loss') :
    
    D = get_distance(out1, out2) 
    E1 = loss1(D)
    
    if map1.shape.as_list()[1:] == [1,1,128] and map2.shape.as_list()[1:] == [1,1,128]:
      map1 = tf.squeeze(map1)
      map2 = tf.squeeze(map2)
      
    E2 = loss2(map1, map2)
#    E3 = loss3(map1, map2)
    
    return tf.add_n([E1,E2], name = "total_loss")
    
def evaluation(features1,features2,labels,thresh):
  with tf.variable_scope('evaluation') :
    distance = tf.sqrt(tf.reduce_sum(tf.square(features1-features2),axis=1))
  
    predicts = tf.cast(tf.greater(distance,thresh),dtype=tf.float32)
    batch_precision = tf.reduce_sum(tf.cast(tf.equal(predicts,labels),dtype=tf.float32))
  
    tp = tf.reduce_sum(tf.cast(tf.equal(tf.add(predicts,labels),2,name='tp'),dtype=tf.float32))
    fn = tf.reduce_sum(tf.cast(tf.equal(tf.add(predicts,labels),0,name='fn'),dtype=tf.float32))
    fp = tf.subtract(tf.reduce_sum(tf.cast(tf.equal(labels,0),dtype=tf.float32)),fn)
    tn = tf.subtract(tf.reduce_sum(tf.cast(tf.equal(labels,1),dtype=tf.float32)),tp)    
    
    eval_all = {'precision':batch_precision,'tp':tp,'tn':tn,'fp':fp,'fn':fn}
  return eval_all  

def training(loss):
  with tf.variable_scope('training') :
    optimizer = tf.train.AdamOptimizer(1e-4, 0.5)

    gen_grads_and_vars = optimizer.compute_gradients(loss)
    gen_train = optimizer.apply_gradients(gen_grads_and_vars)

    ema = tf.train.ExponentialMovingAverage(decay=0.99)
    update_losses = ema.apply([loss])

    global_step = tf.contrib.framework.get_or_create_global_step()
    incr_global_step = tf.assign(global_step, global_step+1)

  return tf.group(update_losses, incr_global_step, gen_train)



def test_get_features():

  image1 = cv2.resize(cv2.imread("1.png"),(32,32))
  image1 = np.expand_dims(cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY),axis=0)
   
  image2 = cv2.resize(cv2.imread("2.png"),(32,32))
  image2 = np.expand_dims(cv2.cvtColor(image2,cv2.COLOR_BGR2GRAY),axis=0)
 
  images = np.expand_dims(np.concatenate((image1,image2),axis=0),4)
  out = get_features(tf.constant(images,dtype=tf.float32))
  sess = tf.Session()
  sess.run(tf.global_variables_initializer())
  value = sess.run(out)
  aaa = np.square(value)
  bbb = np.sum(aaa,axis=-1)
  return bbb

if __name__ == "__main__":
  a=tf.constant([[1,2],[3,4],[5,6]],dtype=tf.float32)
  a=tf.divide(a,tf.norm(a,axis=0))
  b=tf.constant([[7,8],[9,10],[11,12]],dtype=tf.float32)
  b=tf.divide(b,tf.norm(b,axis=0))
  
  distance = get_distance(a,b)
  
  
  distance1= []
  
  for i in range(2):
    for j in range(2):
      distance1.append(tf.norm(a[:,i]-b[:,j]))
      
  sess = tf.Session()    
  d1,d2 = sess.run([distance,distance1])
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
