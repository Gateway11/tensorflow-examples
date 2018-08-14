import numpy as np
import tensorflow as tf
import cv2
import os
import random
import time
 
number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
data_dir = '/Users/daixiang/deep-learning/tensorflow/data/captcha_image'

width = 160
height = 60
max_captcha = 4
batch_size = 64
num_numbers = len(number)

def get_train_data(data_dir = data_dir):
    simples = {}
    for file_name in os.listdir(data_dir):
        captcha = file_name.split('.')[0]
        simples[data_dir + '/' + file_name] = captcha
    return simples

simples = get_train_data(data_dir)
file_simples = list(simples.keys())
num_simples = len(simples)
 
def test(input_, label_):
    saver = tf.train.import_meta_graph('/Users/daixiang/deep-learning/tensorflow/shutiao/output.model-4100.meta')
    graph = tf.get_default_graph()
    inputs = graph.get_tensor_by_name('input:0')
    predict_max_idx = graph.get_tensor_by_name('predict_max_idx:0')
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint('/Users/daixiang/deep-learning/tensorflow/shutiao'))
        predict = sess.run(predict_max_idx, feed_dict = {inputs:[input_]})
        print(predict[0], label_, predict[0] == label_)

for i in range(num_simples):
    input_ = np.float32(cv2.imread(file_simples[i], 0)).flatten() / 255
    label_ = [ord(captcha) - 48 for captcha in simples[file_simples[i]]]
    test(input_, label_)
    #print(file_simples[i])
