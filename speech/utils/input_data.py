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

    return batches_labels, batches_samples, next_idx

def padding_context(sample, num_contexts):
    train_inputs = np.zeros((sample.shape[0], sample.shape[1] + 2 * sample.shape[1] * num_contexts))
    zeropad = np.zeros((sample.shape[1]))

    # 每一步数据由三部分拼接而成，分为当前样本的前9个序列样本，当前样本序列，后9个序列样本
    for time_slice in range(train_inputs.shape[0]):
        # 前9个序列样本, 不足补0
        num_zero_paddings_past = max(0, (num_contexts - time_slice))
        padding_data_past = list(zeropad for slots in range(num_zero_paddings_past))
        source_data_past = sample[max(0, time_slice - num_contexts) : time_slice]

        # 后9个序列样本, 不足补0
        num_zero_paddings_future = max(0, (time_slice - (sample.shape[0] - num_contexts - 1)))
        padding_data_future = list(zeropad for slots in range(num_zero_paddings_future))
        source_data_future = sample[time_slice + 1 : time_slice + num_contexts + 1]
        
        data_past = source_data_past if not num_zero_paddings_past else np.concatenate((padding_data_past, source_data_past))
        data_future = source_data_future if not num_zero_paddings_future else np.concatenate((padding_data_future, source_data_future))

        data_past = np.reshape(data_past, num_contexts * sample.shape[1])
        data_now = sample[time_slice]
        data_future = np.reshape(data_future, num_contexts * sample.shape[1])

        train_inputs[time_slice] = np.concatenate((data_past, data_now, data_future))

    train_inputs = (train_inputs - np.mean(train_inputs)) / np.std(train_inputs)
    return train_inputs
    

def align_samples(batches_label, batches_sample):
    return batches_label, batches_sample

if __name__ == "__main__":
    sample = np.loadtxt("./exp/B11_258.txt")
    padding_context(sample[::2], 9)
