import time
import tensorflow as tf
from tensorflow.python.ops import ctc_ops

from local.models import *
from utils.label_wav import *
from utils.input_data import *


def train(
        train_sample_files,
        train_vector_labels,
        test_sample_files,
        test_vector_labels,
        num_inputs,
        num_contexts,
        lexicon,
        training_steps,
        learning_rate,
        batch_size,
        summaries_dir,
        train_dir,
        save_step_interval,
        model_architecture,
        model_size_info):

    X = tf.placeholder(dtype=tf.float32, shape=[
        None, None, num_inputs + (2 * num_inputs * num_contexts)], name='input')
    sequence_len = tf.placeholder(dtype=tf.int32, shape=[None], name='sequence_len')
    Y = tf.sparse_placeholder(dtype=tf.int32)

    num_character = len(lexicon) + 1
    model_settings = prepare_model_settings(20, num_character)
    logits, dropout_prob = create_model(
        X, sequence_len, model_settings, model_architecture, model_size_info, True)

    with tf.name_scope('loss'):
        avg_loss = tf.reduce_mean(ctc_ops.ctc_loss(text, logits, sequence_len))
        tf.summary.scalar('loss', avg_loss)
    with tf.name_scope('train'):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(avg_loss)
    with tf.name_scope("decode"):
        decoded, log_prob = ctc_ops.ctc_beam_search_decoder(logits, sequence_len, merge_repeated=False)
    with tf.name_scope("accuracy"):
        evaluation_step = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), text))
        tf.summary.scalar('accuracy', evaluation_step)

    sess = tf.InteractiveSession()
    saver = tf.train.Saver(max_to_keep=1)

    tf.global_variables_initializer().run()
    ckpt = tf.train.latest_checkpoint(train_dir)
    if ckpt is not None: saver.restore(sess, ckpt)

    merged_summaries = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(summaries_dir, sess.graph)

    train_num_batches = len(train_sample_files) // batch_size
    test_num_batches = len(test_sample_files) // batch_size
    for epoch in range(training_steps):
        train_next_idx = 0
        test_next_idx = 0
        epoch_start = time.time()

        for batch in range(train_num_batches):
            sparse_labels, batches_sample, length_seqs, train_next_idx = get_next_batches(
                train_next_idx, train_sample_files, train_vector_labels, num_contexts, batch_size)
            train_summary, loss, _ = sess.run(
                [merged_summaries, avg_loss, optimizer],
                feed_dict={X: batches_sample,
                           Y: sparse_labels,
                           sequence_len: length_seqs,
                           dropout_prob: 0.95})
            train_writer.add_summary(train_summary, batch)
            print(
                'batches: %4d/%d, loss: %f' %
                (batch + 1, train_num_batches, loss))
        for batch in range(test_num_batches):
            sparse_labels, batches_sample, length_seqs, test_next_idx = get_next_batches(
                test_next_idx, test_sample_files, test_vector_labels, num_contexts, batch_size)
            d, evaluation_step = sess.run([decoded[0], evaluation_step], feed_dict={X: batches_sample,
                                                                                    Y: sparse_labels,
                                                                                    sequence_len: length_seqs,
                                                                                    dropout_prob: 1.0})
            print('WER: %.2f%%, training_steps: %d/%d' % (evaluation_step, epoch, training_steps))
            dense_decoded = tf.sparse_tensor_to_dense(d, default_value=-1).eval(session=sess)
            dense_labels = trans_tuple_to_texts(sparse_labels, lexicon)

            for orig, decoded_array in zip(dense_labels, dense_decoded):
                decoded_str = trans_array_to_text(decoded_array, lexicon)
                print('语音原始文本: {}'.format(orig))
                print('识别出来的文本: {}'.format(decoded_str))
                break

        saver.save(sess, train_dir + "speech.model", global_step=epoch)
        epoch_duration = time.time() - epoch_start
