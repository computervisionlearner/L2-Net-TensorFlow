#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 18:59:06 2018

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
  roc_auc = auc(fpr,tpr)

  index = np.argmin(np.abs(tpr-0.95))
  print("fpr95 = {} and auc = {}".format(fpr[index],roc_auc))


def do_eval(sess,
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
    predicts_value = 2 - predicts_value
    outputs.extend(predicts_value)
    labels.extend(label)
    
  draw_roc(outputs,labels)

  
def run_train():
  """Train CAPTCHA for a number of steps."""
  
  test_data = dataset.read_data_sets(dataset_dir = '/home/sw/Documents/rgb-nir2/nirscene1/field_2ch.npz')
  with tf.Graph().as_default():
    
    images_pl1, images_pl2, labels_pl = placeholder_inputs(BATCH_SIZE)
    conv_features1,features1 = model.get_features(images_pl1, reuse = False)
    conv_features2,features2 = model.get_features(images_pl2, reuse = True)
    predicts = tf.sqrt(tf.reduce_sum(tf.square(features1-features2),axis=1))

    saver = tf.train.Saver()
    sess = tf.Session()

    saver.restore(sess, "ckpt/model.ckpt-479000")

    print('Test Data Eval:')
    do_eval(sess,
                  predicts,
                  images_pl1,images_pl2,
                  labels_pl,
                  test_data,name='notredame')


    sess.close()

if __name__ == '__main__':
  run_train()