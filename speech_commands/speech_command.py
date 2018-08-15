import tensorflow as tf
import numpy as np
import os
from collections import Counter
import librosa

wav_path = '/Users/daixiang/deep-learning/tensorflow/data/data_wsj/wav/train'
label_file = '/Users/daixiang/deep-learning/tensorflow/data/data_wsj/doc/trans/train.word.txt'

def get_wav_files(wav_path = wav_path):
    wav_files = []
    for (dirpath, dirnames, filenames) in os.walk(wav_path):
        for filename in filenames:
            if(filename.endswith(".wav") or filename.endswith(".WAV")):
                filename_path = os.sep.join([dirpath, filename])
                if os.stat(filename_path).st_size < 240000:
                    continue
                wav_files.append(filename_path)

    return wav_files

def get_wav_labels(wav_files, label_file = label_file):
    labels_dict = {}
    with open(label_file, "r", encoding = 'utf-8') as f:
        for label in f:
            label = label.strip("\n")
            label_id, label_text = label.split(' ', 1)
            labels_dict[label_id] = label_text
    labels = []
    new_wav_files = []
    for wav_file in wav_files:
        wav_id = os.path.basename(wav_file).split(".")[0]
        if wav_id in labels_dict:
            labels.append(labels_dict[wav_id])
            new_wav_files.append(wav_file)

    return new_wav_files, labels

pointer = 0
def get_next_patchs(batch_size, wav_max_len):
    global pointer
    batches_wavs = []
    batches_labels = []
    for i in range(batch_size):
        mfcc = np.loadtxt(wav_files[pointer].split('.')[0] + '.mfcc')
        batches_wavs.append(mfcc.tolist())
        batches_labels.append(labels_vector[pointer])
        pointer += 1

    # 取零补齐
    # label append 0 , 0 对应的字符
    # mfcc 默认的计算长度为20(n_mfcc of mfcc) 作为channel length
    for mfcc in batches_wavs:
        while len(mfcc) < wav_max_len:
            mfcc.append([0] * 20)
    for label in batches_labels:
        while len(label) < label_max_len:
            label.append(0)

    return batches_wavs, batches_labels

conv1d_index = 0
def conv1d_layer(input_tensor, size, dim, activation, scale, bias):
    global conv1d_index
    with tf.variable_scope("conv1d_" + str(conv1d_index)):
        W = tf.get_variable('W', 
                (size, input_tensor.get_shape().as_list()[-1], dim), 
                dtype = tf.float32, 
                initializer = tf.random_uniform_initializer(minval = -scale, maxval = scale))
        if bias:
            b = tf.get_variable('b', [dim], dtype = tf.float32, initializer = tf.constant_initializer(0))
        out = tf.nn.conv1d(input_tensor, W, stride = 1, padding = 'SAME') + (b if bias else 0)

        if not bias:
            beta = tf.get_variable('beta', dim, dtype = tf.float32, initializer = tf.constant_initializer(0))
            gamma = tf.get_variable('gamma', dim, dtype = tf.float32, initializer = tf.constant_initializer(1))
            mean_running = tf.get_variable('mean', dim, dtype = tf.float32, initializer = tf.constant_initializer(0))
            variance_running = tf.get_variable('variance', dim, dtype = tf.float32, initializer = tf.constant_initializer(1))
            mean, variance = tf.nn.moments(out, axes = list(range(len(out.get_shape()) - 1)))

            def update_running_stat():
                decay = 0.99

                # 定义了均值方差指数衰减 见 http://blog.csdn.net/liyuan123zhouhui/article/details/70698264
                update_op = [mean_running.assign(mean_running * decay + mean * (1 - decay)), 
                        variance_running.assign(variance_running * decay + variance * (1 - decay))]
 
                # 指定先执行均值方差的更新运算 见 http://blog.csdn.net/u012436149/article/details/72084744
                with tf.control_dependencies(update_op):
                    return tf.identity(mean), tf.identity(variance)
 
            # 条件运算(https://applenob.github.io/tf_9.html) 按照作者这里的指定 是不进行指数衰减的
            m, v = tf.cond(tf.Variable(False, trainable = False), update_running_stat, lambda: (mean_running, variance_running))
            out = tf.nn.batch_normalization(out, m, v, beta, gamma, 1e-8)

        if activation == 'tanh':
            out = tf.nn.tanh(out)
        elif activation == 'sigmoid':
            out = tf.nn.sigmoid(out)

        conv1d_index += 1
        return out

# 极黑卷积层 https://www.zhihu.com/question/57414498
# 其输入参数中要包含一个大于 1 的rate 输出 channels与输入相同
aconv1d_index = 0
def aconv1d_layer(input_tensor, size, rate, activation, scale, bias):
    global aconv1d_index
    with tf.variable_scope('aconv1d_' + str(aconv1d_index)):
        shape = input_tensor.get_shape().as_list()
        
        # 利用 2 维极黑卷积函数计算相应 1 维卷积，expand_dims squeeze做了相应维度处理
        # 实际 上一个 tf.nn.conv1d 在之前的tensorflow版本中是没有的，其的一个实现也是经过维度调整后调用 tf.nn.conv2d
        W = tf.get_variable('W', 
                (1, size, shape[-1], shape[-1]),
                dtype = tf.float32, 
                initializer = tf.random_uniform_initializer(minval = -scale, maxval = scale))
        if bias:
            b = tf.get_variable('b', [shape[-1]], dtype = tf.float32, initializer = tf.constant_initializer(0))
        out = tf.nn.atrous_conv2d(tf.expand_dims(input_tensor, dim = 1), W, rate = rate, padding = 'SAME')
        out = tf.squeeze(out, [1])

        if not bias:
            beta = tf.get_variable('beta', shape[-1], dtype = tf.float32, initializer = tf.constant_initializer(0))
            gamma = tf.get_variable('gamma', shape[-1], dtype = tf.float32, initializer = tf.constant_initializer(1))
            mean_running = tf.get_variable('mean', shape[-1], dtype = tf.float32, initializer = tf.constant_initializer(0))
            variance_running = tf.get_variable('variance', shape[-1], dtype = tf.float32, initializer = tf.constant_initializer(1))
            mean, variance = tf.nn.moments(out, axes = list(range(len(out.get_shape()) - 1)))

            def update_running_stat():
                decay = 0.99
                update_op = [mean_running.assign(mean_running * decay + mean * (1 - decay)), 
                        variance_running.assign(variance_running * decay + variance * (1 - decay))]
                with tf.control_dependencies(update_op):
                    return tf.identity(mean), tf.identity(variance)
 
            m, v = tf.cond(tf.Variable(False, trainable = False), update_running_stat, lambda: (mean_running, variance_running))
            out = tf.nn.batch_normalization(out, m, v, beta, gamma, 1e-8)

        if activation == 'tanh':
            out = tf.nn.tanh(out)
        elif activation == 'sigmoid':
            out = tf.nn.sigmoid(out)

        aconv1d_index += 1
        return out

def speech_to_text_network(n_dim = 128, n_blocks = 3):
    out = conv1d_layer(input_tensor = X, size = 1, dim = n_dim, activation = 'tanh', scale = 0.14, bias = False)

    def residual_block(input_sensor, size, rate):
        conv_filter = aconv1d_layer(input_tensor = input_sensor, 
                size = size, 
                rate = rate, 
                activation = 'tanh', 
                scale = 0.03, 
                bias = False)
        conv_gate = aconv1d_layer(input_tensor = input_sensor,
                size = size,
                rate = rate,
                activation = 'sigmoid',
                scale = 0.03,
                bias = False)
        out = conv_filter * conv_gate
        out = conv1d_layer(out, size = 1, dim = n_dim, activation = 'tanh', scale = 0.08, bias = False)
        return out + input_sensor, out

    skip = 0
    for _ in range(n_blocks):
        for r in [1, 2, 4, 8, 16]:
            out, s = residual_block(out, size = 7, rate = r)
            skip += s

    logit = conv1d_layer(skip, size = 1, dim = skip.get_shape().as_list()[-1], activation = 'tanh', scale = 0.08, bias = False)

    # 最后卷积层输出是词汇表大小
    logit = conv1d_layer(logit, size = 1, dim = words_size, activation = None, scale = 0.04, bias = True)
    return logit

# 作者自己定义了优化器
class MaxPropOptimizer(tf.train.Optimizer):
    def __init__(self, learning_rate = 0.001, beta2 = 0.999, use_locking = False, name = "MaxProp"):
        super(MaxPropOptimizer, self).__init__(use_locking, name)
        self._lr = learning_rate
        self._beta2 = beta2
        self._lr_t = None
        self._beta2_t = None
    def _prepare(self):
        self._lr_t = tf.convert_to_tensor(self._lr, name = "learning_rate")
        self._beta2_t = tf.convert_to_tensor(self._beta2, name = "beta2")
    def _create_slots(self, var_list):
        for v in var_list:
            self._zeros_slot(v, "m", self._name)
    def _apply_dense(self, grad, var):
        lr_t = tf.cast(self._lr_t, var.dtype.base_dtype)
        beta2_t = tf.cast(self._beta2_t, var.dtype.base_dtype)
        if var.dtype.base_dtype == tf.float16:
            eps = 1e-7
        else:
            eps = 1e-8
        m = self.get_slot(var, "m")
        m_t = m.assign(tf.maximum(beta2_t * m + eps, tf.abs(grad)))
        g_t = grad / m_t
        var_update = tf.assign_sub(var, lr_t * g_t)
        return tf.group(*[var_update, m_t])
    def _apply_sparse(self, grad, var):
        return self._apply_dense(grad, var)

def train_speech_to_text_network(wav_max_len):
    logit = speech_to_text_network()

    # CTC loss
    indices = tf.where(tf.not_equal(tf.cast(Y, tf.float32), 0.))
    target = tf.SparseTensor(indices = indices, values = tf.gather_nd(Y, indices) - 1, dense_shape = tf.cast(tf.shape(Y), tf.int64))
    loss = tf.nn.ctc_loss(target, logit, sequence_len, time_major = False)
    # optimizer
    lr = tf.Variable(0.001, dtype = tf.float32, trainable = False)
    optimizer = MaxPropOptimizer(learning_rate = lr, beta2 = 0.99)
    var_list = [t for t in tf.trainable_variables()]
    gradient = optimizer.compute_gradients(loss, var_list = var_list)
    optimizer_op = optimizer.apply_gradients(gradient)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables())

        for epoch in range(16):
            sess.run(tf.assign(lr, 0.001 * (0.97 ** epoch)))

            global pointer
            pointer = 0
            for batch in range(n_batch):
                batches_wavs, batches_labels = get_next_patchs(batch_size, wav_max_len)
                train_loss, _ = sess.run([loss, optimizer_op], feed_dict = {X: batches_wavs, Y:batches_labels})
                print(epoch, batch, train_loss)
            if epoch % 5 == 0:
                saver.save(sess, r'/Users/daixiang/deep-learning/tensorflow/shutiao/speech/module', global_step = epoch)

def speech_to_text(wav_file):
    wav, sr = librosa.load(wav_file, sr = None, mono = True)
    mfcc = np.transpose(np.expand_dims(librosa.feature.mfcc(wav, sr), axis = 0), [0, 2, 1])

    logit = speech_to_text_network()

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint('.'))

        decoded = tf.transpose(logit, perm = [1, 0, 2])
        decoded, _ = tf.nn.ctc_beam_search_decoder(decoded, sequence_len, merge_repeated = False)
        predict = tf.sparse_to_dense(decoded[0].indices, decoded[0].shape, decoded[0].values) + 1
        output = sess.run(decoded, feed_dict = {X: mfcc})
        print(output)


if __name__ == "__main__":
    wav_files = get_wav_files()
    wav_files, labels = get_wav_labels(wav_files = wav_files)
    print(u"样本数 ：", len(wav_files))
    
    all_words = []
    for label in labels:
        # 字符分解
        all_words += [word for word in label]

    counter = Counter(all_words)
    counter_pairs = sorted(counter.items(), key = lambda x: -x[1])

    words, _ = zip(*counter_pairs)
    words_size = len(words)
    print(u"词汇表大小：", words_size)

    word_num_map = dict(zip(words, range(len(words))))

    # 当字符不在已经收集的words中时，赋予其应当的num，这是一个动态的结果
    to_num = lambda word: word_num_map.get(word, len(words))
 
    # 将单个file的标签映射为num 返回对应list,最终all file组成嵌套list
    labels_vector = [list(map(to_num, label)) for label in labels]
    
    # 将单个file的标签映射为num 返回对应list,最终all file组成嵌套list
    labels_vector = [list(map(to_num, label)) for label in labels]

    label_max_len = np.max([len(label) for label in labels_vector])
    print(u"最长句子的字数:" + str(label_max_len))
    
    wav_max_len = 673
#    for wav_name in wav_files:
#        wav, sr = librosa.load(wav_name, sr = None, mono = True)
#        mfcc = np.transpose(librosa.feature.mfcc(wav, sr), [1, 0])
#        np.savetxt(wav_name.split('.')[0] + '.mfcc', mfcc)
#        if len(mfcc) > wav_max_len:
#            wav_max_len = len(mfcc)
    print("最长的语音", wav_max_len)

    batch_size = 16
    n_batch = len(wav_files) // batch_size

    X = tf.placeholder(dtype = tf.float32, shape = [batch_size, None, 20])

    # 实际mfcc中的元素并非同号，不严格的情况下如此得到序列长度也是可行的
    sequence_len = tf.reduce_sum(tf.cast(tf.not_equal(tf.reduce_sum(X, reduction_indices = 2), 0.), tf.int32), reduction_indices = 1)

    Y = tf.placeholder(dtype = tf.int32, shape = [batch_size, None])

    train_speech_to_text_network(wav_max_len)
