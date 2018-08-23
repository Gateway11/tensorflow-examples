import tensorflow as tf

def prepare_model_settings(relu_clip, num_character):
    return {
        'relu_clip': relu_clip,
        'num_character': num_character,
    }

def create_model(data_input, model_settings, model_architecture, model_size_info, is_training):
  if model_architecture == 'birnn':
    return create_birnn_model(fingerprint_input, model_size_info, is_training)
  else:
    raise Exception('model_architecture argument "' + model_architecture +
                    '" not recognized, should be one of "birnn"')

def load_variables_from_checkpoint(sess, start_checkpoint):
  saver = tf.train.Saver(tf.global_variables())
  saver.restore(sess, start_checkpoint)

def create_birnn_model(data_input, model_settings, model_size_info, is_training):
    """Builds a model with a birnn layer (with output projection layer and peep-hole connections)
    """
    if is_training:
        dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
    
    relu_clip = model_settings['relu_clip']
    num_character = model_settings['num_character']
    w_stddev = 0.046875
    h_stddev = 0.046875

    input_shape = tf.shape(data_input)
    data_input = tf.transpose(data_input, [1, 0, 2])
    data_input = tf.reshape(data_input, [-1, input_shape[-1]])

    with tf.name_scope('Layer_1'):
        W = self.variable_on_device('W', [input_shape[-1], model_size_info[0]], tf.random_normal_initializer(stddev = h_stddev))
        b = self.variable_on_device('b', [model_size_info[0]], tf.random_normal_initializer(stddev = b_stddev))
        layer1_output = tf.minimum(tf.nn.relu(tf.add(tf.matmul(data_input, W), b)), relu_clip)
        layer1_output = tf.nn.dropout(layer1_output, dropout_prob)

    with tf.name_scope('Layer_2'):
        W = self.variable_on_device('W', [model_size_info[0], model_size_info[1]], tf.random_normal_initializer(stddev = h_stddev))
        b = self.variable_on_device('b', [model_size_info[1]], tf.random_normal_initializer(stddev = b_stddev))
        layer2_output = tf.minimum(tf.nn.relu(tf.add(tf.matmul(layer1_output, W), b)), relu_clip)
        layer2_output = tf.nn.dropout(layer2_output, dropout_prob)

    with tf.name_scope('Layer_3'):
        W = self.variable_on_device('W', [model_size_info[1], model_size_info[2]], tf.random_normal_initializer(stddev = h_stddev))
        b = self.variable_on_device('b', [model_size_info[2]], tf.random_normal_initializer(stddev = b_stddev))
        layer3_output = tf.minimum(tf.nn.relu(tf.add(tf.matmul(layer2_output, W), b)), relu_clip)
        layer3_output = tf.nn.dropout(layer3_output, dropout_prob)

    with tf.name_scope('Layer_4'):
        lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(model_size_info[4], forget_bias = 1.0, state_is_tuple = True)
        lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(lstm_fw_cell, input_keep_prob = dropout_prob)

        lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(model_size_info[4], forget_bias = 1.0, state_is_tuple = True)
        lstm_bw_cell = tf.contrib.rnn.DropoutWrapper(lstm_bw_cell, input_keep_prob = dropout_prob)

        layer3_output = tf.reshape(layer3_output, [-1, input_shape[0], model_size_info[3]])
        layer4_output, output_states = tf.nn.bidirectional_dynamic_rnn(cell_fw = lstm_fw_cell,
                                                                 cell_bw = lstm_bw_cell,
                                                                 inputs = layer3_output,
                                                                 dtype = tf.float32,
                                                                 time_major = True,
                                                                 sequence_length = seq_length)

        layer4_output = tf.concat(layer4_output, 2)
        layer4_output = tf.reshape(outputs, [-1, 2 * model_size_info[4]])

    with tf.name_scope('Layer_5'):
        W = self.variable_on_device('W', [model_size_info[4], model_size_info[5]], tf.random_normal_initializer(stddev = h_stddev))
        b = self.variable_on_device('b', [model_size_info[5]], tf.random_normal_initializer(stddev = b_stddev))
        layer5_output = tf.minimum(tf.nn.relu(tf.add(tf.matmul(layer4_output, W), b)), relu_clip)
        layer5_output = tf.nn.dropout(layer5_output, dropout_prob)

    with tf.name_scope('Layer_6'):
        W = self.variable_on_device('W', [model_size_info[5], num_character], tf.random_normal_initializer(stddev = h_stddev))
        b = self.variable_on_device('b', [num_character], tf.random_normal_initializer(stddev = b_stddev))
        layer6_output = tf.add(tf.matmul(layer5_output, h6), b6)

    output = tf.reshape(layer6_output, [-1, input_shape[0], num_character])
    return output, dropout_prob if is_training else output

def variable_on_device(name, shape, initializer, use_gpu = False):
    if use_gpu:
        with tf.device('/gpu:0'):
            var = tf.get_variable(name = name, shape = shape, initializer = initializer)
    else:
        var = tf.get_variable(name = name, shape = shape, initializer = initializer)
    return var

if __name__ == '__main__':
    input_tensor = tf.placeholder(tf.float32, [None, None, 494], name = 'input')  # 语音log filter bank or MFCC features
    text = tf.sparse_placeholder(tf.int32, name = 'text')  # 文本
    seq_length = tf.placeholder(tf.int32, [None], name = 'seq_length')  # 序列长
