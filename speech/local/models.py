import tensorflow as tf

def prepare_model_settings(relu_clip, num_character):
    return {
        'relu_clip': relu_clip,
        'num_character': num_character,
    }

def create_model(data_input, seq_length, model_settings, model_architecture, model_size_info, is_training):
    if model_architecture == 'birnn':
        return create_birnn_model(data_input, seq_length, model_settings, model_size_info, is_training)
    else:
        raise Exception('model_architecture argument "' + model_architecture +
                        '" not recognized, should be one of "birnn"')

def load_variables_from_checkpoint(sess, start_checkpoint):
    saver = tf.train.Saver(tf.global_variables())
    saver.restore(sess, start_checkpoint)

def create_birnn_model(data_input, seq_length, model_settings, model_size_info, is_training):
    """Builds a model with a birnn layer (with output projection layer and peep-hole connections)
    """
    if is_training:
        dropout_prob = tf.placeholder(tf.float32, name = 'dropout_prob')
    
    relu_clip = model_settings['relu_clip']
    num_character = model_settings['num_character']
    w_stddev = 0.046875
    b_stddev = 0.046875

    input_shape = tf.shape(data_input) # shape(8, 137, 494)
    num_inputs = data_input.shape[-1]
    # 时间序列优先
    data_input = tf.transpose(data_input, [1, 0, 2]) 
    data_input = tf.reshape(data_input, [-1, num_inputs]) # shape(num_step * batch_size, 494)

    with tf.name_scope('Layer_1'):
        W = variable_on_device('W1', [num_inputs, model_size_info[0]], tf.random_normal_initializer(stddev = w_stddev))
        b = variable_on_device('b1', [model_size_info[0]], tf.random_normal_initializer(stddev = b_stddev))
        layer1_output = tf.minimum(tf.nn.relu(tf.add(tf.matmul(data_input, W), b)), relu_clip)
        if is_training:
            layer1_output = tf.nn.dropout(layer1_output, dropout_prob)

    with tf.name_scope('Layer_2'):
        W = variable_on_device('W2', [model_size_info[0], model_size_info[1]], tf.random_normal_initializer(stddev = w_stddev))
        b = variable_on_device('b2', [model_size_info[1]], tf.random_normal_initializer(stddev = b_stddev))
        layer2_output = tf.minimum(tf.nn.relu(tf.add(tf.matmul(layer1_output, W), b)), relu_clip)
        if is_training:
            layer2_output = tf.nn.dropout(layer2_output, dropout_prob)

    with tf.name_scope('Layer_3'):
        W = variable_on_device('W3', [model_size_info[1], model_size_info[2]], tf.random_normal_initializer(stddev = w_stddev))
        b = variable_on_device('b3', [model_size_info[2]], tf.random_normal_initializer(stddev = b_stddev))
        layer3_output = tf.minimum(tf.nn.relu(tf.add(tf.matmul(layer2_output, W), b)), relu_clip)
        if is_training:
            layer3_output = tf.nn.dropout(layer3_output, dropout_prob)

    with tf.name_scope('Layer_4'):
        # 前向
        lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(model_size_info[3], forget_bias = 1.0, state_is_tuple = True)
        if is_training:
            lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(lstm_fw_cell, input_keep_prob = dropout_prob)

        # 后向
        lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(model_size_info[3], forget_bias = 1.0, state_is_tuple = True)
        if is_training:
            lstm_bw_cell = tf.contrib.rnn.DropoutWrapper(lstm_bw_cell, input_keep_prob = dropout_prob)

        layer3_output = tf.reshape(layer3_output, [-1, input_shape[0], model_size_info[2]])
        layer4_output, output_states = tf.nn.bidirectional_dynamic_rnn(cell_fw = lstm_fw_cell,
                                                                 cell_bw = lstm_bw_cell,
                                                                 inputs = layer3_output,
                                                                 dtype = tf.float32,
                                                                 time_major = True,
                                                                 sequence_length = seq_length)
        layer4_output = tf.concat(layer4_output, 2)
        layer4_output = tf.reshape(layer4_output, [-1, 2 * model_size_info[3]])

    with tf.name_scope('Layer_5'):
        W = variable_on_device('W5', [2 * model_size_info[3], model_size_info[4]], tf.random_normal_initializer(stddev = w_stddev))
        b = variable_on_device('b5', [model_size_info[4]], tf.random_normal_initializer(stddev = b_stddev))
        layer5_output = tf.minimum(tf.nn.relu(tf.add(tf.matmul(layer4_output, W), b)), relu_clip)
        if is_training:
            layer5_output = tf.nn.dropout(layer5_output, dropout_prob)

    with tf.name_scope('Layer_6'):
        W = variable_on_device('W6', [model_size_info[4], num_character], tf.random_normal_initializer(stddev = w_stddev))
        b = variable_on_device('b6', [num_character], tf.random_normal_initializer(stddev = b_stddev))
        layer6_output = tf.add(tf.matmul(layer5_output, W), b)

    output = tf.reshape(layer6_output, [-1, input_shape[0], num_character])
    return output, dropout_prob if is_training else output

def variable_on_device(name, shape, initializer, use_gpu = False):
    if use_gpu:
        with tf.device('/gpu:0'):
            var = tf.get_variable(name = name, shape = shape, initializer = initializer)
    else:
        var = tf.get_variable(name = name, shape = shape, initializer = initializer)
    return var
