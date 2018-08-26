import numpy as np
from utils.label_wav import sparse_tuple_from


def get_next_batches(next_idx, sample_files, sample_labels, num_contexts, batch_size):
    batch_labels = []
    batch_samples = []
    for i in range(batch_size):
        sample = np.loadtxt(sample_files[next_idx + i])
        sample = padding_context(sample[::2], num_contexts)

        batch_samples.append(sample.astype('float32'))
        batch_labels.append(sample_labels[next_idx + i])

    batch_samples, num_steps = align_batch_samples(batch_samples)
    sparse_labels = sparse_tuple_from(batch_labels)

    return sparse_labels, batch_samples, num_steps 


def padding_context(sample, num_contexts):
    num_input = sample.shape[1]  # 26
    train_inputs = np.zeros((sample.shape[0], num_input + 2 * num_input * num_contexts))  # shape(417, 494)
    zeropad = np.zeros((num_input))

    # 每一步数据由三部分拼接而成，分为当前样本的前9个序列样本，当前样本序列，后9个序列样本
    for time_slice in range(train_inputs.shape[0]):
        # 前9个序列样本, 不足补0
        num_zero_paddings_past = max(0, (num_contexts - time_slice))
        padding_data_past = [zeropad for slots in range(num_zero_paddings_past)]
        source_data_past = sample[max(0, time_slice - num_contexts): time_slice]

        # 后9个序列样本, 不足补0
        num_zero_paddings_future = max(0, (time_slice - (sample.shape[0] - num_contexts - 1)))
        padding_data_future = [zeropad for slots in range(num_zero_paddings_future)]
        source_data_future = sample[time_slice + 1: time_slice + num_contexts + 1]

        data_past = source_data_past if not num_zero_paddings_past else np.concatenate(
            (padding_data_past, source_data_past))
        data_future = source_data_future if not num_zero_paddings_future else np.concatenate(
            (source_data_future, padding_data_future))

        data_past = np.reshape(data_past, num_contexts * num_input)
        data_now = sample[time_slice]
        data_future = np.reshape(data_future, num_contexts * num_input)

        train_inputs[time_slice] = np.concatenate((data_past, data_now, data_future))

    train_inputs = (train_inputs - np.mean(train_inputs)) / np.std(train_inputs)
    return train_inputs


def align_batch_samples(batch_samples, dtype=np.float32, value=0.):
    batch_size = len(batch_samples)
    num_steps = [len(sample) for sample in batch_samples]
    max_step = np.max(num_steps)
    num_input = batch_samples[0].shape[1]

    train_inputs = (np.ones((batch_size, max_step, num_input)) * value).astype(dtype)  # shape(8, 468, 494)
    for idx, sample in enumerate(batch_samples):
        train_inputs[idx, :len(sample)] = np.asarray(sample)

    return train_inputs, num_steps 


if __name__ == "__main__":
    samples = []
    samples.append(padding_context(np.loadtxt('../output/mfcc/train/B11_258.txt')[::2], 9).astype('float32'))
    samples.append(padding_context(np.loadtxt('../output/mfcc/train/A4_211.txt')[::2], 9).astype('float32'))
    align_samples(samples)
