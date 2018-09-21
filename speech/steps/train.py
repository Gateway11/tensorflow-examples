import random
import time
import tensorflow as tf
from tensorflow.python.ops import ctc_ops

from local.models import *
from utils.label_wav import *
from utils.input_data import *


def train(audio_processer, num_inputs, num_classes, model_architecture, model_size_info, 
        learning_rate, training_steps, batch_size, aligning, eval_step_interval, output_dir):

    X = tf.placeholder(dtype=tf.float32, shape=[
        None, audio_processer.get_max_step(aligning), num_inputs], name='input_tensor')
    sequence_len = tf.placeholder(dtype=tf.int32, shape=[None], name='sequence_len')
    Y = tf.sparse_placeholder(dtype=tf.int32, name='output_tensor')

    model_settings = prepare_model_settings(20, num_classes)
    logits, dropout_prob = create_model(
        X, sequence_len, model_settings, model_architecture, model_size_info, True)

    with tf.name_scope('loss'):
        avg_loss = tf.reduce_mean(ctc_ops.ctc_loss(Y, logits, sequence_len))
        tf.summary.scalar('loss', avg_loss)
    with tf.name_scope('train'):
        learning_rate_input = tf.placeholder(tf.float32, [], name='learning_rate_input')
        train_step = tf.train.AdamOptimizer(learning_rate=learning_rate_input).minimize(avg_loss)
    with tf.name_scope("decoder"):
        decoder, _ = ctc_ops.ctc_beam_search_decoder(logits, sequence_len, merge_repeated=False)
    with tf.name_scope("accuracy"):
        evaluation_step = tf.reduce_mean(tf.edit_distance(tf.cast(decoder[0], tf.int32), Y))
        tf.summary.scalar('accuracy', evaluation_step)

    if tf.test.gpu_device_name():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    else:
        sess = tf.InteractiveSession()
    saver = tf.train.Saver(max_to_keep=1)
    sess.run(tf.global_variables_initializer())
    ckpt = tf.train.latest_checkpoint(output_dir + 'train/')
    if ckpt: saver.restore(sess, ckpt)

    merged_summaries = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(output_dir + 'train/logs/', sess.graph)

    num_train_batches = audio_processer.get_batch_count(batch_size, 'train')
    num_dev_batches = audio_processer.get_batch_count(batch_size, 'dev')

    total_training_step = sum(training_steps)
    for training_step in range(total_training_step):
        total_train_loss = 0
        epoch_start = time.time()

        learning_rate_value = learning_rate[1 if training_step > training_steps[0] else 0]
        for train_batch in range(num_train_batches):
            train_data = audio_processer.get_data(train_batch * batch_size, batch_size, 'train', aligning)
            #train_summary, loss, _ = sess.run([merged_summaries, avg_loss, train_step],
            loss, _ = sess.run([avg_loss, train_step], 
                    feed_dict={X: train_data[0], Y: train_data[1], sequence_len: \
                            train_data[2], learning_rate_input: learning_rate_value, dropout_prob: 0.95})
            #train_writer.add_summary(train_summary, train_batch)
            total_train_loss += loss

        time_cost = time.time() - epoch_start
        print('training step: %d/%d, train loss: %g, time cost: %.2fs' 
                % (training_step + 1, total_training_step, total_train_loss / num_train_batches, time_cost))

        if (training_step + 1) % eval_step_interval == 0:
            saver.save(sess, output_dir + "train/speech-model.ckpt", global_step=training_step)

            rand_batch = random.randint(0, num_dev_batches - 1)
            dev_data = audio_processer.get_data(rand_batch * batch_size, batch_size, 'dev', aligning)
            dev_accuracy = sess.run(evaluation_step, 
                    feed_dict={X: dev_data[0], Y: dev_data[1], sequence_len: dev_data[2], dropout_prob: 1.0})
            print('WER: %.2f, training step: %d/%d' % (dev_accuracy, training_step + 1, total_training_step))

    total_test_accuracy = 0
    num_test_batches = audio_processer.get_batch_count(batch_size, 'test')
    for test_batch in range(num_test_batches):
        test_data = audio_processer.get_data(test_batch * batch_size, batch_size, 'test', aligning)
        decodes, accuracy = sess.run([decoder[0], evaluation_step],
            feed_dict={X: test_data[0], Y: test_data[1], sequence_len: test_data[2], dropout_prob: 1.0})

        total_test_accuracy += accuracy
        dense_decodes = tf.sparse_tensor_to_dense(decodes, default_value=-1).eval(session=sess)
        dense_labels = trans_tuple_to_texts(test_data[1], lexicon)
        for orig, decode_array in zip(dense_labels, dense_decodes):
            decoded_str = trans_array_to_text(decode_array, lexicon)
            print('语音原始文本: {}'.format(orig))
            print('识别出来的文本: {}'.format(decoded_str))
            break

    print('Final WER: %.2f, train steps: %d' % (total_test_accuracy, total_training_step))
