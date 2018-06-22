#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 31 16:53:05 2018

@author: sw
"""

import numpy as np
import time
import model
import dataset
import tensorflow as tf
import os
from datetime import datetime
import logging
from read_record import Reader
import sys

from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve,auc

def view_bar(message, num, total):
    rate = num / total
    rate_num = int(rate * 40)
    rate_nums = np.ceil(rate * 100)
    r = '\r%s:[%s%s]%d%%\t%d/%d' % (message, ">" * rate_num, " " * (40 - rate_num), rate_nums, num, total,)
    sys.stdout.write(r)
    sys.stdout.flush()

BATCH_SIZE = 128

IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64
checkpoint_dir = 'ckpt'
checkpoint_file = os.path.join(checkpoint_dir, 'model.ckpt')
train_dir='summary'

def initLogging(logFilename='record.log'):
  """Init for logging
  """
  logging.basicConfig(
                    level    = logging.DEBUG,
                    format='%(asctime)s-%(levelname)s-%(message)s',
                    datefmt  = '%y-%m-%d %H:%M',
                    filename = logFilename,
                    filemode = 'w');
  console = logging.StreamHandler()
  console.setLevel(logging.INFO)
  formatter = logging.Formatter('%(asctime)s-%(levelname)s-%(message)s')
  console.setFormatter(formatter)
  logging.getLogger('').addHandler(console)
initLogging()



def placeholder_inputs(batch_size):
  images_pl1 = tf.placeholder(tf.float32,
                        shape=(batch_size,IMAGE_HEIGHT,IMAGE_WIDTH,1))

  images_pl2 = tf.placeholder(tf.float32,
                        shape=(batch_size,IMAGE_HEIGHT,IMAGE_WIDTH,1))
  
  labels_pl = tf.placeholder(tf.int32, shape=(batch_size,))
  return images_pl1, images_pl2, labels_pl

def fill_feed_dict(data_set, images_pl1, images_pl2, labels_pl, shuffle = True):

  images_feed, labels_feed = data_set.next_batch(BATCH_SIZE,shuffle = shuffle)

  images1,images2 =np.split(images_feed,2,axis=3)
  feed_dict = {
      images_pl1: images1,
      images_pl2: images2,
      labels_pl: labels_feed,
  }
  return feed_dict, labels_feed

def draw_roc(outputs,labels):
  fpr,tpr,thresh = roc_curve(labels,outputs)
  
  idx = np.argmin(np.abs(tpr-0.95))
  logging.info("fpr95 = {}".format(fpr[idx]))  
  roc_auc = auc(fpr,tpr)
  plt.plot(fpr, tpr, lw=1, label='AUC = %0.4f' %  roc_auc)
  plt.xlim([0, 1])  
  plt.ylim([0.6, 1])  
  plt.xlabel('False Positive Rate')  
  plt.ylabel('True Positive Rate') 
  plt.title('ROC curve')  
  plt.legend(loc="lower right")  
  plt.savefig('roc_water_2ch.png')
  np.savez('roc_water_2ch',fpr,tpr)
  plt.show()

def do_eval(sess,steps,
            predicts,
            images_pl1,images_pl2,
            labels_pl,
            data_set,name):
  """Runs one evaluation against the full epoch of data.

  Args:
    sess: The session in which the model has been trained.
    eval_correct: The Tensor that returns the number of correct predictions.
    images_placeholder: The images placeholder.
    labels_placeholder: The labels placeholder.
    data_set: The set of images and labels to evaluate, from
      input_data.read_data_sets().
  """
  # And run one epoch of eval.
  outputs = []
  labels = []
  steps_per_epoch = data_set.num_examples // BATCH_SIZE
  num_examples = steps_per_epoch * BATCH_SIZE
  
  for step in range(steps_per_epoch):
    feed_dict, label = fill_feed_dict(data_set,
                               images_pl1,images_pl2,
                               labels_pl,shuffle=False)
    predicts_value = sess.run(predicts, feed_dict=feed_dict)

    outputs.extend(predicts_value)
    labels.extend(label)
    
  precision = np.mean(np.equal(labels,np.greater(outputs,0.5)))
  draw_roc(outputs,labels,steps,name)
  logging.info('\r Num examples: %d    Precision @ 1: %.4f------------' %(num_examples, precision))
  
def run_test():
  """Train CAPTCHA for a number of steps."""
  test_data = dataset.read_data_sets(dataset_dir = '/home/sw/Documents/rgb-nir2/nirscene1/water_2ch.npz')
  with tf.Graph().as_default():
    
    images_pl1, images_pl2, labels_pl = placeholder_inputs(BATCH_SIZE)
    conv_features1,features1 = model.get_features(images_pl1, reuse = False)
    conv_features2,features2 = model.get_features(images_pl2, reuse = True)
    predicts = tf.sqrt(tf.reduce_sum(tf.square(features1-features2),axis=1))
    
    saver = tf.train.Saver()
    sess = tf.Session()
    saver.restore(sess, "ckpt/model.ckpt-479000")

    outputs = []
    labels = []
    
    steps_per_epoch = test_data.num_examples // BATCH_SIZE
    num_examples = steps_per_epoch * BATCH_SIZE
    
    for step in range(steps_per_epoch):
      feed_dict, label = fill_feed_dict(test_data,images_pl1,images_pl2,labels_pl,shuffle=False)
      predicts_value = sess.run(predicts,feed_dict=feed_dict)
      predicts_value = 2-predicts_value
      outputs.extend(predicts_value)
      labels.extend(label)

      view_bar('processing:', step, steps_per_epoch)

    draw_roc(outputs,labels)
    sess.close()

if __name__ == '__main__':
  run_test()
