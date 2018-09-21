#!/usr/bin/env python3

import argparse
import os
import re
import sys

from steps.decoder import *
from steps.train import *
from tensorflow.python.platform import app
from utils.download_and_untar import *
from utils.input_data import *

FLAGS = None


def main(_):
    # 下载数据集，默认下载清华数据集
    maybe_download_and_untar(FLAGS.data_url.split(','), FLAGS.data_dir)
    # 扫描数据集，提取MFCC特征, 生成字典, label向量化
    audio_processer = AudioPorcesser(FLAGS.data_dir, FLAGS.num_filters, 
            FLAGS.downsampling_ratio, FLAGS.num_contexts, FLAGS.output_dir)

    lexicon = audio_processer.prepare(os.path.basename(FLAGS.data_url).split('.')[0])
    with open(FLAGS.output_dir + 'symbol_table.txt', 'w') as f:
        f.write(re.sub('[\s\'{}]', '', str(lexicon)).replace(',', '\n').replace(':', '\t'))

    if FLAGS.model_architecture == 'birnn':
        num_inputs = FLAGS.num_filters + 2 * FLAGS.num_filters * FLAGS.num_contexts
    else:
        num_inputs = FLAGS.num_filters
    # 开始训练
    train(audio_processer, num_inputs, len(lexicon) + 1,
            FLAGS.model_architecture, FLAGS.model_size_info, 
            FLAGS.learning_rate, FLAGS.training_steps, FLAGS.batch_size, 
            FLAGS.aligning, FLAGS.eval_step_interval, FLAGS.output_dir)

    #decoder(audio_processer, FLAGS.output_dir + 'train/', 1, 'BATCH', lexicon)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_url',
        type=str,
        default='http://www.openslr.org/resources/18/data_thchs30.tgz',
        help='Location of speech training data archive on the web.')
    parser.add_argument(
        '--data_dir',
        type=str,
        default='./data/thchs30/',
        help='Where to download the speech training data to.')
    parser.add_argument(
        '--num_filters',
        type=int,
        default=26,
        help='How many bins to use for the MFCC fingerprint.')
    parser.add_argument(
        '--downsampling_ratio',
        type=int,
        default=2,
        help='What is the ratio of the downsampling.')
    parser.add_argument(
        '--num_contexts',
        type=int,
        default=9,
        help='How much information is there before and after each step.')
    parser.add_argument(
        '--training_steps',
        type=int,
        nargs="+",
        default=[60, 20],
        help='How many training loops to run')
    parser.add_argument(
        '--eval_step_interval',
        type=int,
        default=5,
        help='How often to evaluate the training results.')
    parser.add_argument(
        '--learning_rate',
        type=float,
        nargs="+",
        default=[0.001, 0.0001],
        help='How large a learning rate to use when training.')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16,
        help='How many items to train with at once.')
    parser.add_argument(
        '--model_architecture',
        type=str,
        default='birnn',
        help='What model architecture to use.')
    parser.add_argument(
        '--model_size_info',
        type=int,
        nargs="+",
        default=[512, 512, 1024, 512, 512],
        help='Model dimensions - different for various models.')
    parser.add_argument(
        '--aligning',
        type=str,
        default='BATCH', #['BATCH', 'MAP']
        help='How to you align data.')
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./output/',
        help='Directory to write event logs and checkpoint.')

    FLAGS, unparsed = parser.parse_known_args()
    app.run(main=main, argv=[sys.argv[0]] + unparsed)
