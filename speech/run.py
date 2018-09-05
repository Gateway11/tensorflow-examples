#!/usr/bin/env python3

import argparse
import sys

from tensorflow.python.platform import app
from utils.label_wav import *
from utils.download_and_untar import *
from steps.train import *

FLAGS = None


def main(_):
    # 下载数据集，默认下载清华数据集
    download_and_untar(FLAGS.data_url.split(','), FLAGS.data_dir)

    # 扫描训练集
    train_wav_files = list_wav_file(FLAGS.data_dir + 'data_thchs30/train')
    train_labels_dict = load_label_file(FLAGS.data_dir + 'resource/trans/train.syllable.txt')

    # 提取MFCC特征, 生成字典, label向量化
    train_sample_files = preapre_wav_list(train_wav_files, FLAGS.dct_coefficient_count, FLAGS.mfcc_dir + 'train/')
    lexicon, train_labels, train_sample_files = prepare_label_list(train_sample_files, train_labels_dict)
    train_vector_labels = labels_to_vector(train_labels, lexicon)

    test_wav_files = list_wav_file(FLAGS.data_dir + 'data_thchs30/test')
    test_labels_dict = load_label_file(FLAGS.data_dir + 'resource/trans/test.syllable.txt')

    test_sample_files = preapre_wav_list(test_wav_files, FLAGS.dct_coefficient_count, FLAGS.mfcc_dir + 'test/')
    lexicon, test_labels, test_sample_files = prepare_label_list(test_sample_files, test_labels_dict)
    test_vector_labels = labels_to_vector(test_labels, lexicon)

    # 开始训练
    train(
        train_sample_files,
        train_vector_labels,
        test_sample_files,
        test_vector_labels,
        lexicon,
        FLAGS.dct_coefficient_count,
        FLAGS.num_contexts,
        FLAGS.how_many_training_steps,
        FLAGS.learning_rate,
        FLAGS.batch_size,
        FLAGS.summaries_dir,
        FLAGS.train_dir,
        FLAGS.eval_step_interval,
        FLAGS.model_architecture,
        FLAGS.model_size_info)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_url',
        type=str,
        # pylint: disable=line-too-long
        default='http://www.openslr.org/resources/18/data_thchs30.tgz,',
        # pylint: enable=line-too-long
        help='Location of speech training data archive on the web.')
    parser.add_argument(
        '--data_dir',
        type=str,
        default='./data/thchs30/',
        help='Where to download the speech training data to.')
    parser.add_argument(
        '--background_volume',
        type=float,
        default=0.1,
        help='How loud the background noise should be, between 0 and 1.')
    parser.add_argument(
        '--background_frequency',
        type=float,
        default=0.8,
        help='How many of the training samples have background noise mixed in.')
    parser.add_argument(
        '--silence_percentage',
        type=float,
        default=10.0,
        help='How much of the training data should be silence.')
    parser.add_argument(
        '--unknown_percentage',
        type=float,
        default=10.0,
        help='How much of the training data should be unknown words.')
    parser.add_argument(
        '--time_shift_ms',
        type=float,
        default=100.0,
        help='Range to randomly shift the training audio by in time.')
    parser.add_argument(
        '--testing_percentage',
        type=int,
        default=10,
        help='What percentage of wavs to use as a test set.')
    parser.add_argument(
        '--validation_percentage',
        type=int,
        default=10,
        help='What percentage of wavs to use as a validation set.')
    parser.add_argument(
        '--sample_rate',
        type=int,
        default=16000,
        help='Expected sample rate of the wavs',)
    parser.add_argument(
        '--clip_duration_ms',
        type=int,
        default=1000,
        help='Expected duration in milliseconds of the wavs',)
    parser.add_argument(
        '--window_size_ms',
        type=float,
        default=30.0,
        help='How long each spectrogram timeslice is.',)
    parser.add_argument(
        '--window_stride_ms',
        type=float,
        default=10.0,
        help='How far to move in time between spectogram timeslices.',)
    parser.add_argument(
        '--dct_coefficient_count',
        type=int,
        default=26,
        help='How many bins to use for the MFCC fingerprint',)
    parser.add_argument(
        '--mfcc_dir',
        type=str,
        default='./output/mfcc/',
        help='Where to save MFCC fingerprint for samples.')
    parser.add_argument(
        '--num_contexts',
        type=int,
        default=9,
        help='How much information is there before and after each step.',)
    parser.add_argument(
        '--how_many_training_steps',
        type=int,
        default=120,
        help='How many training loops to run',)
    parser.add_argument(
        '--eval_step_interval',
        type=int,
        default=15,
        help='How often to evaluate the training results.')
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.001,
        help='How large a learning rate to use when training.')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=8,
        help='How many items to train with at once',)
    parser.add_argument(
        '--summaries_dir',
        type=str,
        default='./output/train/logs/',
        help='Where to save summary logs for TensorBoard.')
    parser.add_argument(
        '--train_dir',
        type=str,
        default='./output/train/',
        help='Directory to write event logs and checkpoint.')
    parser.add_argument(
        '--save_step_interval',
        type=int,
        default=1,
        help='Save model checkpoint every save_steps.')
    parser.add_argument(
        '--start_checkpoint',
        type=str,
        default='./output/train',
        help='If specified, restore this pretrained model before any training.')
    parser.add_argument(
        '--model_architecture',
        type=str,
        default='birnn',
        help='What model architecture to use')
    parser.add_argument(
        '--model_size_info',
        type=int,
        nargs="+",
        default=[512, 512, 1024, 512, 512],
        help='Model dimensions - different for various models')
    parser.add_argument(
        '--check_nans',
        type=bool,
        default=False,
        help='Whether to check for invalid numbers during processing')

    FLAGS, unparsed = parser.parse_known_args()
    app.run(main=main, argv=[sys.argv[0]] + unparsed)
