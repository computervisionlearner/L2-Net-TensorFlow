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

def draw_roc(outputs,labels,step,name):
  fpr,tpr,thresh = roc_curve(labels,outputs)
  roc_auc = auc(fpr,tpr)
#  plt.plot(fpr, tpr, lw=1, label='AUC = %0.4f' %  roc_auc)
#  plt.xlim([0, 0.2])  
#  plt.ylim([0, 1])  
#  plt.xlabel('False Positive Rate')  
#  plt.ylabel('True Positive Rate') 
#  plt.title('ROC curve')  
#  plt.legend(loc="lower right")  
#  plt.savefig('roc_{}_{}.png'.format(name,step))
  index = np.argmin(np.abs(tpr-0.95))
  logging.info("name = {} step = {} fpr95 = {} and auc = {}".format(name,step,fpr[index],roc_auc))
#  plt.show()

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
  
def run_train():
  """Train CAPTCHA for a number of steps."""
  
  with tf.Graph().as_default():
    train_reader = Reader('/home/sw/Documents/rgb-nir2/qd_fang2_9_8/country_2ch.tfrecord',name='train_data',batch_size=BATCH_SIZE)
    leftIMG, rightIMG, labels_op = train_reader.feed() #[64,128]
    

    conv_features1,features1 = model.get_features(leftIMG, reuse = False)
    conv_features2,features2 = model.get_features(rightIMG, reuse = True)
    predicts = tf.sqrt(tf.reduce_sum(tf.square(features1-features2),axis=1))
    
    total_loss = model.caculate_loss(conv_features1, conv_features2, features1, features2)   
    eval_all = model.evaluation(features1, features2, labels_op, 1)   #train
    
    tf.summary.scalar('sum_loss', total_loss)
    tf.summary.scalar('roc/tp',eval_all['tp'])
    tf.summary.scalar('roc/fp',eval_all['fp'])
    tf.summary.scalar('roc/tpr',eval_all['tp']/(eval_all['tp']+eval_all['fn']))
    tf.summary.scalar('roc/precision',eval_all['precision'])
    train_op = model.training(total_loss)
    
    summary = tf.summary.merge_all()
    saver = tf.train.Saver(max_to_keep=50)
    init_op = tf.global_variables_initializer()

    sess = tf.Session()
    summary_writer = tf.summary.FileWriter(train_dir, sess.graph)
    sess.run(init_op)
#    saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir))

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    
    try:
      max_step = 100000
      for step in range(1,max_step):
        start_time = time.time()
        _, summary_str, loss_value, predicts_value, f1, f2 = sess.run([train_op, summary, total_loss, predicts, features1, features2])
        summary_writer.add_summary(summary_str, step)
        summary_writer.flush()
        duration = time.time() - start_time
        if step % 10 == 0:
          logging.info('\r>> Step %d run_train: loss = %.4f  (%.3f sec)'
                % (step, loss_value, duration))
          #-------------------------------
        if step % 1000 == 0:
          logging.info('>> %s Saving in %s' % (datetime.now(), checkpoint_dir))
          saver.save(sess, checkpoint_file, global_step=step)


    except KeyboardInterrupt:
        print('INTERRUPTED')
        coord.request_stop()

    finally:
        saver.save(sess, checkpoint_file, global_step=step)
        print('\rModel saved in file :%s'%checkpoint_dir)
        coord.request_stop()
        coord.join(threads)

    sess.close()

if __name__ == '__main__':
  run_train()
