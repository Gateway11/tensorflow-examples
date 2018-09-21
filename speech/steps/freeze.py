import tensorflow as tf

def freeze(nnet_path):
    saver.restore(sess, tf.train.latest_checkpoint(nnet_path))
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint(nnet_path))
        tf.train.write_graph(sess.graph_def, nnet_path, 'output_graph.pb')
