

def prepare_model_settings(relu_clip, num_character):
    return {
        'relu_clip': relu_clip,
        'num_character': num_character,
    }


def create_model(input_tensor, sequence_len, 
        model_settings, model_architecture, model_size_info, is_training):
    if model_architecture == 'birnn':
        return create_birnn_model(input_tensor, 
                sequence_len, model_settings, model_size_info, is_training)
    else:
        raise Exception('model_architecture argument "' + model_architecture +
                        '" not recognized, should be one of "birnn"')


def create_birnn_model(input_tensor, 
        sequence_len, model_settings, model_size_info, is_training):
    if is_training:
        dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')

    relu_clip = model_settings['relu_clip']
    num_character = model_settings['num_character']
    w_stddev = 0.046875
    b_stddev = 0.046875

    input_shape = tf.shape(input_tensor)  # shape(8, 137, 494)
    num_inputs = input_tensor.shape[-1]
    # 时间序列优先
    input_tensor = tf.transpose(input_tensor, [1, 0, 2])
    # shape(num_step * batch_size, 494)
    input_tensor = tf.reshape(input_tensor, [-1, num_inputs])

    with tf.name_scope('Layer_1'):
        W = variable_on_device('W1', [num_inputs, model_size_info[0]],
                               tf.random_normal_initializer(stddev=w_stddev))
        b = variable_on_device('b1', [model_size_info[0]],
                               tf.random_normal_initializer(stddev=b_stddev))
        outputs = tf.minimum(tf.nn.relu(tf.add(tf.matmul(input_tensor, W), b)), relu_clip)
        if is_training:
            outputs = tf.nn.dropout(outputs, dropout_prob)

    with tf.name_scope('Layer_2'):
        W = variable_on_device('W2', [model_size_info[0], model_size_info[1]],
                               tf.random_normal_initializer(stddev=w_stddev))
        b = variable_on_device('b2', [model_size_info[1]],
                               tf.random_normal_initializer(stddev=b_stddev))
        outputs = tf.minimum(tf.nn.relu(tf.add(tf.matmul(outputs, W), b)), relu_clip)
        if is_training:
            outputs = tf.nn.dropout(outputs, dropout_prob)

    with tf.name_scope('Layer_3'):
        W = variable_on_device('W3', [model_size_info[1], model_size_info[2]],
                               tf.random_normal_initializer(stddev=w_stddev))
        b = variable_on_device('b3', [model_size_info[2]],
                               tf.random_normal_initializer(stddev=b_stddev))
        outputs = tf.minimum(tf.nn.relu(tf.add(tf.matmul(outputs, W), b)), relu_clip)
        if is_training:
            outputs = tf.nn.dropout(outputs, dropout_prob)

    with tf.name_scope('Layer_4'):
        # 前向
        lstm_fw = tf.contrib.rnn.BasicLSTMCell(
            model_size_info[3], forget_bias=1.0, state_is_tuple=True)
        if is_training:
            lstm_fw = tf.contrib.rnn.DropoutWrapper(lstm_fw, input_keep_prob=dropout_prob)

        # 后向
        lstm_bw = tf.contrib.rnn.BasicLSTMCell(
            model_size_info[3], forget_bias=1.0, state_is_tuple=True)
        if is_training:
            lstm_bw = tf.contrib.rnn.DropoutWrapper(lstm_bw, input_keep_prob=dropout_prob)

        outputs = tf.reshape(outputs, [-1, input_shape[0], model_size_info[2]])
        outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_fw, cell_bw=lstm_bw, 
                inputs=outputs, dtype=tf.float32, time_major=True, sequence_length=sequence_len)
        outputs = tf.reshape(tf.concat(outputs, 2), [-1, 2 * model_size_info[3]])

    with tf.name_scope('Layer_5'):
        W = variable_on_device('W5', [2 * model_size_info[3], model_size_info[4]],
                               tf.random_normal_initializer(stddev=w_stddev))
        b = variable_on_device('b5', [model_size_info[4]],
                               tf.random_normal_initializer(stddev=b_stddev))
        outputs = tf.minimum(tf.nn.relu(tf.add(tf.matmul(outputs, W), b)), relu_clip)
        if is_training:
            outputs = tf.nn.dropout(outputs, dropout_prob)

    with tf.name_scope('Layer_6'):
        W = variable_on_device('W6', [model_size_info[4], num_character],
                               tf.random_normal_initializer(stddev=w_stddev))
        b = variable_on_device('b6', [num_character],
                               tf.random_normal_initializer(stddev=b_stddev))
        outputs = tf.add(tf.matmul(outputs, W), b)

    outputs = tf.reshape(layer6_outputs, [-1, input_shape[0], num_character])
    return outputs, dropout_prob if is_training else outputs


def variable_on_device(name, shape, initializer, use_gpu=False):
    if use_gpu:
        with tf.device('/gpu:0'):
            var = tf.get_variable(
                name=name, shape=shape, initializer=initializer)
    else:
        var = tf.get_variable(name=name, shape=shape, initializer=initializer)
    return var
