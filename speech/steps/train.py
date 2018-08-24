import time
import tensorflow as tf
from tensorflow.python.ops import ctc_ops

from local.models import *
from utils.label_wav import *
from utils.input_data import *

def train(train_sample_files, train_vector_labels, test_sample_files, test_vector_labels,
        num_inputs, num_contexts, lexicon, training_steps, learning_rate, batch_size, 
        summaries_dir, train_dir, save_step_interval, model_architecture, model_size_info):
    
    input_tensor = tf.placeholder(tf.float32, 
            [None, None, num_inputs + (2 * num_inputs * num_contexts)], name='input')
    text = tf.sparse_placeholder(tf.int32, name='text')  # 文本
    seq_length = tf.placeholder(tf.int32, [None], name='seq_length')  # 序列长

    num_character = len(lexicon) + 1
    model_settings  = prepare_model_settings(20, num_character)
    logits, dropout_prob = create_model(input_tensor, 
            seq_length, model_settings, model_architecture, model_size_info, True)

    with tf.name_scope('loss'): # 损失
        avg_loss = tf.reduce_mean(ctc_ops.ctc_loss(text, logits, seq_length))
        tf.summary.scalar('loss', avg_loss)
    # [optimizer]
    with tf.name_scope('train'): # 训练过程
        optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(avg_loss)

    with tf.name_scope("decode"):
        decoded, log_prob = ctc_ops.ctc_beam_search_decoder(logits, seq_length, merge_repeated = False)

    with tf.name_scope("accuracy"):
        distance = tf.edit_distance(tf.cast(decoded[0], tf.int32), text)
        # 计算label error rate (accuracy)
        label_err = tf.reduce_mean(distance, name = 'label_error_rate')
        tf.summary.scalar('accuracy', label_err)

    saver = tf.train.Saver(max_to_keep = 1)  # 生成saver
    sess = tf.Session()
    # 没有模型的话，就重新初始化
    sess.run(tf.global_variables_initializer())

    ckpt = tf.train.latest_checkpoint(train_dir)
    startepo = 0
    if ckpt != None:
        saver.restore(sess, ckpt)
        ind = ckpt.rfind("-")
        startepo = int(ckpt[ind + 1:])

    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter(summaries_dir, sess.graph)

    train_num_batches = int(np.ceil(len(train_sample_files) / batch_size))
    test_num_batches = int(np.ceil(len(test_sample_files) / batch_size))
    for epoch in range(training_steps):
        train_next_idx = 0
        epoch_start = time.time()

        for batch in range(train_num_batches):
            sparse_labels, batches_sample, length_seqs, train_next_idx = get_next_batches(train_next_idx, 
                                                                                        train_sample_files, 
                                                                                        train_vector_labels, 
                                                                                        num_contexts, 
                                                                                        batch_size)
            train_loss, _ = sess.run([avg_loss, optimizer], feed_dict = {input_tensor:batches_sample, 
                                                                        text:sparse_labels, 
                                                                        seq_length:length_seqs, 
                                                                        dropout_prob:0.95})
            print('batches: %4d/%d, train_loss: %f' % (batch, train_num_batches, train_loss))
        if epoch % save_step_interval == 0:
            test_next_idx = 0
            total_wer = 0
            for batch in range(test_num_batches):
                sparse_labels, batches_sample, length_seqs, test_next_idx = get_next_batches(test_next_idx, 
                                                                                        test_sample_files, 
                                                                                        test_vector_labels, 
                                                                                        num_contexts, 
                                                                                        batch_size)
                d, wer = sess.run([decoded[0], label_err], feed_dict = {input_tensor:batches_sample, 
                                                                        text:sparse_labels, 
                                                                        seq_length:length_seqs, 
                                                                        dropout_prob:1.0})
                total_wer += wer

            print('WER: %.2f%%, training_steps: %d/%d' % (total_wer / test_num_batches, epoch, training_steps))
            dense_decoded = tf.sparse_tensor_to_dense(d, default_value = -1).eval(session = sess)
            dense_labels = trans_tuple_to_texts(sparse_labels, lexicon)

            for orig, decoded_array in zip(dense_labels, dense_decoded):
                decoded_str = trans_array_to_text(decoded_array, lexicon)
                print('语音原始文本: {}'.format(orig))
                print('识别出来的文本: {}'.format(decoded_str))
                break

            saver.save(sess, train_dir + "speech.model", global_step = epoch)
            epoch_duration = time.time() - epoch_start

    sess.close()
