import tensorflow as tf


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
    if model_architecture == 'fsmn':
        return create_fsmn_model(input_tensor, 
                sequence_len, model_settings, model_size_info, is_training)
    if model_architecture == 'dfcnn':
        return create_dfcnn_model(input_tensor, 
                sequence_len, model_settings, model_size_info, is_training)
    else:
        raise Exception('model_architecture argument "' + model_architecture +
                '" not recognized, should be one of "birnn", "fsmn", "dfcnn"')

def load_variables_from_checkpoint(sess, start_checkpoint):
    saver = tf.train.Saver(tf.global_variables())
    saver.restore(sess, start_checkpoint)


def create_birnn_model(input_tensor, 
        sequence_len, model_settings, model_size_info, is_training):
    if is_training:
        dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')

    relu_clip = model_settings['relu_clip']
    num_character = model_settings['num_character']
    stddev = 0.046875

    # shape(batch_size, num_steps, 494)
    input_shape_tensor = tf.shape(input_tensor)  
    num_inputs = input_tensor.shape[-1]
    # 时间序列优先
    input_tensor = tf.transpose(input_tensor, [1, 0, 2])
    # shape(num_steps * batch_size, 494)
    input_tensor = tf.reshape(input_tensor, [-1, num_inputs])

    outputs = fc_layer('FC_1', input_tensor, [num_inputs, model_size_info[0]], stddev, 'relu')
    outputs = tf.minimum(outputs, relu_clip)
    if is_training:
        outputs = tf.nn.dropout(outputs, dropout_prob)

    outputs = fc_layer('FC_2', outputs, [model_size_info[0], model_size_info[1]], stddev, 'relu')
    outputs = tf.minimum(outputs, relu_clip)
    if is_training:
        outputs = tf.nn.dropout(outputs, dropout_prob)

    outputs = fc_layer('FC_3', outputs, [model_size_info[1], model_size_info[2]], stddev, 'relu')
    outputs = tf.minimum(outputs, relu_clip)
    if is_training:
        outputs = tf.nn.dropout(outputs, dropout_prob)

    with tf.name_scope('Layer_4'):
        # 前向
        lstm_fw = tf.contrib.rnn.BasicLSTMCell(model_size_info[3], forget_bias=1.0, state_is_tuple=True)
        if is_training:
            lstm_fw = tf.contrib.rnn.DropoutWrapper(lstm_fw, input_keep_prob=dropout_prob)
        # 后向
        lstm_bw = tf.contrib.rnn.BasicLSTMCell(model_size_info[3], forget_bias=1.0, state_is_tuple=True)
        if is_training:
            lstm_bw = tf.contrib.rnn.DropoutWrapper(lstm_bw, input_keep_prob=dropout_prob)

        outputs = tf.reshape(outputs, [-1, input_shape_tensor[0], model_size_info[2]])
        outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_fw, cell_bw=lstm_bw, 
                inputs=outputs, dtype=tf.float32, time_major=True, sequence_length=sequence_len)
        outputs = tf.reshape(tf.concat(outputs, 2), [-1, 2 * model_size_info[3]])

    outputs = fc_layer('FC_5', outputs, [model_size_info[3] * 2, model_size_info[4]], stddev, 'relu')
    outputs = tf.minimum(outputs, relu_clip)
    if is_training:
        outputs = tf.nn.dropout(outputs, dropout_prob)

    outputs = fc_layer('FC_6', outputs, [model_size_info[4], num_character], stddev, None)
    outputs = tf.reshape(outputs, [-1, input_shape_tensor[0], num_character])
    return outputs, dropout_prob if is_training else outputs


def create_fsmn_model(input_tensor, 
        sequence_len, model_settings, model_size_info, is_training):
    if is_training:
        dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')

    num_character = model_settings['num_character']
    input_shape_tensor = tf.shape(input_tensor)  
    num_inputs = input_tensor.shape[-1]

    outputs = fsmn_layer(input_tensor, [512, 512], 10, 'fsmn1')
    outputs = fsmn_layer(outputs, [512, 1024], 10, 'fsmn2')

    outputs = fc_layer('FC_3', input_tensor, [1024, num_character], stddev, None)
    return outputs, dropout_prob if is_training else outputs


def create_dfcnn_model(input_tensor, 
        sequence_len, model_settings, model_size_info, is_training):
    pass


def fsmn_layer(input_tensor, shape, mem_size, name):
    with tf.variable_scope(name):
        W1 = tf.get_variable("W1", shape, initializer=tf.truncated_normal_initializer(stddev=5e-2, dtype=tf.float32))
        W2 = tf.get_variable("W2", shape, initializer=tf.truncated_normal_initializer(stddev=5e-2, dtype=tf.float32))
        b = tf.get_variable("b", shape[-1], initializer = tf.constant_initializer(0.0, dtype=tf.float32))
        mem = tf.get_variable("mem", [mem_size], initializer = tf.constant_initializer(1.0, dtype=tf.float32))

        #num_steps = input_tensor.get_shape()[1].value
        input_shape = tf.shape(input_tensor)  
        num_steps = input_shape.eval()[1] 
        memory_matrix = []
        for step in range(num_steps):
            left_num = tf.maximum(0, step + 1 - mem_size) # 0 ~ 29
            right_num = num_steps - step - 1 # 29 ~ 0

            mem = mem[tf.minimum(step, mem_size)::-1]

            d_batch = tf.pad(mem, [[left_num, right_num]])
            memory_matrix.append([d_batch])

        memory_matrix = tf.concat(memory_matrix, 0)

        batch_size = input_data.get_shape()[0].value
        h_hatt = tf.matmul([memory_matrix] * batch_size, input_data)
        return tf.matmul(input_data, [W1] * batch_size) + tf.add(tf.matmul(h_hatt, [W2] * batch_size), b)


def fc_layer(name, input_tensor, shape, stddev, activation):
    with tf.variable_scope(name):
        W = tf.get_variable('W', shape, initializer=tf.random_normal_initializer(stddev=stddev))
        b = tf.get_variable('b', shape[-1], initializer=tf.random_normal_initializer(stddev=stddev))
        outputs = tf.add(tf.matmul(input_tensor, W), b)

        if activation == 'relu':
            outputs = tf.nn.relu(outputs)

    return outputs
