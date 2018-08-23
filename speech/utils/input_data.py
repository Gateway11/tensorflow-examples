import numpy as np

def get_next_batches(next_idx, files, labels, num_contexts, batch_size):
    batches_label = []
    batches_sample = []
    for i in range(batch_size):
        sample = np.loadtxt(files[next_idx])
        sample = padding_context(sample[::2], num_contexts)

        batches_sample.append(sample.astype('float32'))
        batches_label.append(labels[next_idx])
        next_idx += 1

    batches_label, batches_sample = align_samples(batches_label, batches_sample)

    return batches_label, batches_sample, next_idx

def padding_context(sample, num_contexts):
    num_input = sample.shape[1] # 26
    train_inputs = np.zeros((sample.shape[0], num_input + 2 * num_input * num_contexts)) # shape(417, 494)
    zeropad = np.zeros((num_input))

    # 每一步数据由三部分拼接而成，分为当前样本的前9个序列样本，当前样本序列，后9个序列样本
    for time_slice in range(train_inputs.shape[0]):
        # 前9个序列样本, 不足补0
        num_zero_paddings_past = max(0, (num_contexts - time_slice))
        padding_data_past = [zeropad for slots in range(num_zero_paddings_past)]
        source_data_past = sample[max(0, time_slice - num_contexts) : time_slice]

        # 后9个序列样本, 不足补0
        num_zero_paddings_future = max(0, (time_slice - (sample.shape[0] - num_contexts - 1)))
        padding_data_future = [zeropad for slots in range(num_zero_paddings_future)]
        source_data_future = sample[time_slice + 1 : time_slice + num_contexts + 1]
        
        data_past = source_data_past if not num_zero_paddings_past else np.concatenate((padding_data_past, source_data_past))
        data_future = source_data_future if not num_zero_paddings_future else np.concatenate((source_data_future, padding_data_future))

        data_past = np.reshape(data_past, num_contexts * num_input)
        data_now = sample[time_slice]
        data_future = np.reshape(data_future, num_contexts * num_input)

        train_inputs[time_slice] = np.concatenate((data_past, data_now, data_future))

    train_inputs = (train_inputs - np.mean(train_inputs)) / np.std(train_inputs)
    return train_inputs
    
def align_samples(batches_label, batches_sample):
    batch_size = len(batches_sample)
    max_step = np.max([len(sample) for sample in batches_sample])
    num_input = batches_sample[0].shape[1]

    train_batches = np.zeros((batch_size, max_step, num_input)).astype(np.float32) # shape(8, 468, 494)
    for idx, sample in enumerate(batches_sample):
        train_batches[idx, :len(sample)] = np.asarray(sample)

    return batches_label, train_batches

def sparse_tuple_from(sequences, dtype=np.int32):
    indices = []
    values = []

    for n, seq in enumerate(sequences):
        indices.extend(zip([n] * len(seq), range(len(seq))))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), indices.max(0)[1] + 1], dtype=np.int64)

    return indices, values, shape

if __name__ == "__main__":
    samples = []
    samples.append(padding_context(np.loadtxt("./exp/B11_258.txt")[::2], 9).astype('float32'))
    samples.append(padding_context(np.loadtxt("./exp/A4_211.txt")[::2], 9).astype('float32'))
    align_samples("xxxx", samples)
