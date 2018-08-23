import tensorflow as tf

def prepare_model_settings(relu_clip, label_count, w_stddev, b_stddev):
    return {
        'relu_clip': relu_clip,
        'label_count': label_count,
        'w_stddev': w_stddev,
        'b_stddev': b_wtddev
    }

def create_model(data_input, model_settings, model_architecture, model_size_info, is_training):
  if model_architecture == 'birnn':
    return create_birnn_model(fingerprint_input, model_size_info, is_training)
  else:
    raise Exception('model_architecture argument "' + model_architecture +
                    '" not recognized, should be one of "birnn"')

def load_variables_from_checkpoint(sess, start_checkpoint):
  """Utility function to centralize checkpoint restoration.

  Args:
    sess: TensorFlow session.
    start_checkpoint: Path to saved checkpoint on disk.
  """
  saver = tf.train.Saver(tf.global_variables())
  saver.restore(sess, start_checkpoint)

def create_birnn_model(data_input, model_settings, model_size_info, is_training):
    """Builds a model with a birnn layer (with output projection layer and peep-hole connections)
    """
    if is_training:
        dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
    
    relu_clip = model_settings['relu_clip']
    label_count = model_settings['label_count']
    w_stddev = model_settings['w_stddev']
    b_stddev = model_settings['b_stddev']

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

    layer3_output = tf.reshape(layer3_output, [-1, input_shape[0], model_size_info[1]])
    with tf.name_scope('Layer_4'):
        lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(model_size_info[4], forget_bias = 1.0, state_is_tuple = True)
        lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(lstm_fw_cell, input_keep_prob = dropout_prob)

        lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(model_size_info[4], forget_bias = 1.0, state_is_tuple = True)
        lstm_bw_cell = tf.contrib.rnn.DropoutWrapper(lstm_bw_cell, input_keep_prob = dropout_prob)

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
        W = self.variable_on_device('W', [model_size_info[5], model_size_info[6]], tf.random_normal_initializer(stddev = h_stddev))
        b = self.variable_on_device('b', [model_size_info[6]], tf.random_normal_initializer(stddev = b_stddev))
        layer6_output = tf.add(tf.matmul(layer5_output, h6), b6)

    output = tf.reshape(layer6_output, [-1, input_shape[0], label_count])
    return output, dropout_prob if is_training else output

def variable_on_device(name, shape, initializer, use_gpu = False):
    if use_gpu:
        with tf.device('/gpu:0'):
            var = tf.get_variable(name = name, shape = shape, initializer = initializer)
    else:
        var = tf.get_variable(name = name, shape = shape, initializer = initializer)
    return var
