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
 
def get_next_batch():
    batch_x = np.zeros([batch_size, width * height])
    batch_y = np.zeros([batch_size, num_numbers * max_captcha])
 
    for i in range(batch_size):
        file_name = file_simples[random.randint(0, num_simples - 1)]
        batch_x[i, :] = np.float32(cv2.imread(file_name, 0)).flatten() / 255
        batch_y[i, :] = text2vec(simples[file_name])
    return batch_x, batch_y

def text2vec(text):
    return [0 if ord(i) - 48 != j else 1 for i in text for j in range(num_numbers)]
####################################################################

x  = tf.placeholder(tf.float32, [None, width * height], name = 'input')
y_ = tf.placeholder(tf.float32, [None, num_numbers * max_captcha])
x_image = tf.reshape(x, [-1, height, width, 1])

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides = [1, 1, 1, 1], padding = 'SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_conv3 = weight_variable([5, 5, 64, 64])
b_conv3 = bias_variable([64])
h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
h_pool3 = max_pool_2x2(h_conv3)

W_fc1 = weight_variable([8 * 20 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool3_flat = tf.reshape(h_pool3, [-1, 8 * 20 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)

W_fc2 = weight_variable([1024, num_numbers * max_captcha])
b_fc2 = bias_variable([num_numbers * max_captcha])
output = tf.add(tf.matmul(h_fc1, W_fc2), b_fc2)

cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = y_, logits = output))
train_step = tf.train.AdamOptimizer(learning_rate = 1e-4).minimize(cross_entropy)

predict = tf.reshape(output, [-1, max_captcha, num_numbers])
labels  = tf.reshape(y_, [-1, max_captcha, num_numbers])
correct_prediction = tf.equal(tf.argmax(predict, 2, name = 'predict_max_idx'), tf.argmax(labels, 2))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

def train():
    sess = tf.InteractiveSession()

    saver = tf.train.Saver()
    tf.global_variables_initializer().run()
    for i in range(5000):
        batch_x, batch_y = get_next_batch()
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict = {x: batch_x, y_: batch_y})
            print("step %d, training accuracy %g " % (i, train_accuracy))
            if train_accuracy > 0.99:
                saver.save(sess, "/Users/daixiang/deep-learning/tensorflow/shutiao/output.model", global_step = i)
        train_step.run(feed_dict = {x: batch_x, y_: batch_y})

train()
