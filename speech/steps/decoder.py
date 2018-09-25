import random
import tensorflow as tf

from utils.label_wav import *

def decoder(audio_processer, nnet_path, lexicon):
    batch_size = 1

    saver = tf.train.import_meta_graph(nnet_path + 'speech-model.ckpt-44.meta')
    graph = tf.get_default_graph()
    
    input_tensor = graph.get_tensor_by_name('input_tensor:0')
    sequence_len = graph.get_tensor_by_name('sequence_len:0')
    dropout_prob = graph.get_tensor_by_name('dropout_prob:0')
    decoder = graph.get_tensor_by_name('decoder/CTCBeamSearchDecoder:1')
    
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint(nnet_path))

        num_test_batches = audio_processer.get_batch_count(batch_size, 'test')
        for test_batch in range(num_test_batches):
            data = audio_processer.get_data(test_batch * batch_size, batch_size, 'test', 'BATCH')
            decodes = sess.run(decoder, feed_dict = {input_tensor:data[0], sequence_len:data[2], dropout_prob:1.0})
            #dense_decodes = tf.sparse_tensor_to_dense(decodes, default_value=-1).eval(session=sess)
            #decoded_str = trans_array_to_text(decode_array, lexicon)
            decoded_str = vector_to_string(decodes, lexicon)
            dense_labels = trans_tuple_to_texts(data[1], lexicon)
            print('语音原始文本: {}'.format(dense_labels))
            print('识别出来的文本: {}'.format(decoded_str))
