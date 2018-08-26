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
        lexicon,
        num_inputs,
        num_contexts,
        training_steps,
        learning_rate,
        batch_size,
        summaries_dir,
        train_dir,
        eval_step_interval,
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
        avg_loss = tf.reduce_mean(ctc_ops.ctc_loss(Y, logits, sequence_len))
        tf.summary.scalar('loss', avg_loss)
    with tf.name_scope('train'):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(avg_loss)
    with tf.name_scope("decoder"):
        decoder, _ = ctc_ops.ctc_beam_search_decoder(logits, sequence_len, merge_repeated=False)
    with tf.name_scope("accuracy"):
        evaluation_step = tf.reduce_mean(tf.edit_distance(tf.cast(decoder[0], tf.int32), Y))
        tf.summary.scalar('accuracy', evaluation_step)

    sess = tf.InteractiveSession()
    saver = tf.train.Saver(max_to_keep=1)

    tf.global_variables_initializer().run()
    ckpt = tf.train.latest_checkpoint(train_dir)
    if ckpt is not None: saver.restore(sess, ckpt)

    merged_summaries = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(summaries_dir, sess.graph)

    num_train_batches = len(train_sample_files) // batch_size
    num_test_batches = len(test_sample_files) // batch_size

    for training_step in range(training_steps):
        for train_batch in range(num_train_batches):
            sparse_labels, batch_samples, num_steps = get_next_batches(
                batch_size * train_batch, train_sample_files, train_vector_labels, num_contexts, batch_size)

            # train_summary, loss, _ = sess.run([merged_summaries, avg_loss, optimizer],
            loss, _ = sess.run([avg_loss, optimizer],
                feed_dict={X: batch_samples, Y: sparse_labels, sequence_len: num_steps, dropout_prob: 0.95})
            # train_writer.add_summary(train_summary, train_batch)
            print('training batch: %4d/%d, loss: %f' % (train_batch + 1, num_train_batches, loss))

        total_wer = 0
        if (training_step + 1) % eval_step_interval == 0:
            for test_batch in range(num_test_batches):
                sparse_labels, batch_samples, num_steps = get_next_batches(
                    batch_size * test_batch, test_sample_files, test_vector_labels, num_contexts, batch_size)
    
                decodes, wer = sess.run([decoder[0], evaluation_step],
                    feed_dict={X: batch_samples, Y: sparse_labels, sequence_len: num_steps, dropout_prob: 1.0})
    
                total_wer += wer
                print('WER: %.2f%%, testing batch: %d/%d' % (wer, test_batch + 1, num_test_batches))
    
                dense_decodes = tf.sparse_tensor_to_dense(decodes, default_value=-1).eval(session=sess)
                dense_labels = trans_tuple_to_texts(sparse_labels, lexicon)
    
                for orig, decode_array in zip(dense_labels, dense_decodes):
                    decoded_str = trans_array_to_text(decode_array, lexicon)
                    print('语音原始文本: {}'.format(orig))
                    print('识别出来的文本: {}'.format(decoded_str))
                    break
    
            print('WER: %.2f%%, training step: %d/%d' 
                    % (total_wer / num_test_batches, training_step + 1, training_steps))
            saver.save(sess, train_dir + "speech.model", global_step=training_step)
