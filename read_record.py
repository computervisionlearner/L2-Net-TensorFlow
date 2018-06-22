#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 23:02:02 2018

@author: sw
"""
import tensorflow as tf
import random
import scipy.misc as misc
import numpy as np
class Reader():
  def __init__(self, tfrecords_file, height=64, width=64,
    min_queue_examples=5000, batch_size=128, num_threads=8, name = ''):
    """
    Args:
      tfrecords_file: string, tfrecords file path
      min_queue_examples: integer, minimum number of samples to retain in the queue that provides of batches of examples
      batch_size: integer, number of images per batch
      num_threads: integer, number of preprocess threads
    """
    self.tfrecords_file = tfrecords_file
    self.height = height
    self.width =width
    self.min_queue_examples = min_queue_examples
    self.batch_size = batch_size
    self.num_threads = num_threads
    self.reader = tf.TFRecordReader()
    self.name = name
  def feed(self):
    """
    Returns:
      images: 4D tensor [batch_size, image_width, image_height, image_depth]
    """
    with tf.name_scope(self.name):
      filename_queue = tf.train.string_input_producer([self.tfrecords_file])
      _, serialized_example = self.reader.read(filename_queue)
  
      features = tf.parse_single_example(
        serialized_example,
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'label_raw': tf.FixedLenFeature([], tf.string)
            })
  
      image = tf.decode_raw(features['image_raw'], tf.uint8)
      image.set_shape([self.height * self.width *2])
      image = tf.cast(image, tf.float32) 
      label = tf.decode_raw(features['label_raw'], tf.int32)
      label = tf.reshape(tf.cast(label, tf.float32),shape=(1,))
      reshape_image = tf.reshape(image, [self.height, self.width, 2])   

      reshape_image = tf.image.random_flip_left_right(reshape_image)
      reshape_image = tf.image.random_flip_up_down(reshape_image)
      left,right = tf.split(reshape_image,2,axis=2)
      
      distorted = tf.concat((left,right),axis=1)  #[64,128,1]
      distorted = tf.image.random_flip_left_right(distorted)
      
      result1 = tf.cond(tf.equal(label[0],1),lambda:tf.image.random_brightness(distorted,max_delta=32./255),lambda:distorted)#
      result4 = tf.cond(tf.equal(label[0],1),lambda:tf.image.random_contrast(result1,lower=0.5,upper=1.5),lambda:distorted)
      result4 = result4/tf.reduce_max(result4)
      
      left, right = tf.split(result4, 2, axis = 1)

      lefts, rights, labels = tf.train.shuffle_batch([left,right,label], batch_size=self.batch_size, num_threads=self.num_threads,
            capacity=self.min_queue_examples + 3*self.batch_size, min_after_dequeue=self.min_queue_examples
          )
  
#      tf.summary.image('record_inputs', images)
      labels = tf.squeeze(labels)
    return lefts, rights, labels
  
  def feed_test(self):
    """
    Returns:
      images: 4D tensor [batch_size, image_width, image_height, image_depth]
    """
    with tf.name_scope(self.name):
      filename_queue = tf.train.string_input_producer([self.tfrecords_file])
      _, serialized_example = self.reader.read(filename_queue)
  
      features = tf.parse_single_example(
        serialized_example,
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'label_raw': tf.FixedLenFeature([], tf.string)
            })
  
      image = tf.decode_raw(features['image_raw'], tf.uint8)
      image.set_shape([self.height * self.width *2])
      image = tf.cast(image, tf.float32) 
      label = tf.decode_raw(features['label_raw'], tf.int32)
      label = tf.reshape(tf.cast(label, tf.float32),shape=(1,))
      reshape_image = tf.reshape(image, [self.height, self.width, 2])   


      left,right = tf.split(reshape_image,2,axis=2)  
      distorted = tf.concat((left,right),axis=1)  #[64,128,1]
     
      result4 = distorted/tf.reduce_max(distorted)
      images,labels = tf.train.batch([result4,label], batch_size=self.batch_size, num_threads=self.num_threads,
            capacity=self.min_queue_examples + 3*self.batch_size )
  
      labels = tf.squeeze(labels)
    return images,labels  
  

if __name__ == '__main__':
  train_reader = Reader('liberty/liberty_train.tfrecord',name='test_data')
  images_op, labels_op = train_reader.feed()
  
  
  init = tf.global_variables_initializer()
  sess = tf.Session()
  sess.run(init)
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(sess=sess, coord=coord)
  imgs, labels = sess.run([images_op, labels_op])
  print(np.max(imgs))
  for i in range(len(imgs)):
    misc.imsave('record_test/{}_{}.jpg'.format(i,labels[i]),np.squeeze(imgs[i]))
    
  coord.request_stop()
  coord.join(threads)
  sess.close()  
