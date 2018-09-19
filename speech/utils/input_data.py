import numpy as np
from utils.label_wav import *


class AudioPorcesser(object):
    def __init__(self, data_dir, num_filters, downsampling_ratio, num_contexts, output_dir):
        self.data_dir = data_dir
        self.num_filters = num_filters
        self.num_contexts = num_contexts
        self.downsampling_ratio = downsampling_ratio 
        self.output_dir = output_dir
        self.sample_files_dict = {}
        self.sample_label_dict = {}
        self.lexicon = {}
        self.max_step = None


    def prepare(self, wav_path_name):
        for mode in ['train', 'dev', 'test']:
            # 扫描数据集
            wav_files = list_wav_file(self.data_dir + wav_path_name + '/' + mode)
            labels_dict = load_label_file(self.data_dir + 'resource/trains/' + mode + '.syllable.txt')
            # 提取MFCC特征, 生成字典, label向量化
            mfcc_dir = self.output_dir + 'mfcc/'+ wav_path_name + '/' + mode \
                    + '/' + str(self.num_filters) + '_' + str(self.downsampling_ratio) + '/'
            sample_files, max_step = preapre_wav_list(wav_files, 
                    self.num_filters, self.downsampling_ratio, mfcc_dir)
            lexicon, labels, sample_files = prepare_label_list(sample_files, labels_dict)
            if mode == 'train': 
                self.lexicon = lexicon
                self.max_step = max_step
            labels_vector = labels_to_vector(labels, self.lexicon)
            self.sample_files_dict[mode] = sample_files
            self.sample_label_dict[mode] = labels_vector 

        return self.lexicon


    def get_max_step(self, aligning):
        return self.max_step if aligning == 'MAP' else None


    def get_batch_count(self, batch_size, mode):
        return len(self.sample_files_dict[mode]) // batch_size


    def get_data(self, data_idx, batch_size, mode, aligning):
        batch_label = []
        batch_sample = []
        for i in range(batch_size):
            sample = np.loadtxt(self.sample_files_dict[mode][data_idx + i])
            if self.num_contexts:
                sample = self.padding_context(sample, self.num_contexts)
    
            batch_sample.append(sample.astype('float32'))
            batch_label.append(self.sample_label_dict[mode][data_idx + i])
    
        batch_sample, step_list = self.align_batch_sample(
                batch_sample, self.get_max_step(aligning))
        sparse_label = sparse_tuple_from(batch_label)
    
        return batch_sample, sparse_label, step_list
    
    
    def padding_context(self, sample, num_contexts):
        num_steps = sample.shape[0]
        num_filters = sample.shape[1]  # 26

        zeropad = np.zeros((num_filters))
        padding_output= np.zeros((num_steps, num_filters + 2 * num_filters * num_contexts))
        # 每一步数据由三部分拼接而成，分为当前样本的前9个序列样本，当前样本序列，后9个序列样本
        for time_slice in range(num_steps):
            # 前9个序列样本, 不足补0
            num_zero_paddings_past = max(0, (num_contexts - time_slice))
            padding_data_past = [zeropad for slots in range(num_zero_paddings_past)]
            source_data_past = sample[max(0, time_slice - num_contexts): time_slice]
            # 后9个序列样本, 不足补0
            num_zero_paddings_future = max(0, (time_slice - (num_steps - num_contexts - 1)))
            padding_data_future = [zeropad for slots in range(num_zero_paddings_future)]
            source_data_future = sample[time_slice + 1: time_slice + num_contexts + 1]
    
            data_past = source_data_past if not num_zero_paddings_past else np.concatenate(
                (padding_data_past, source_data_past))
            data_future = source_data_future if not num_zero_paddings_future else np.concatenate(
                (source_data_future, padding_data_future))
    
            data_past = np.reshape(data_past, num_contexts * num_filters)
            data_now = sample[time_slice]
            data_future = np.reshape(data_future, num_contexts * num_filters)
    
            padding_output[time_slice] = np.concatenate((data_past, data_now, data_future))
    
        padding_output = (padding_output - np.mean(padding_output)) / np.std(padding_output)
        return padding_output 
    
    
    def align_batch_sample(self, batch_sample, max_step=None):
        batch_size = len(batch_sample)
        step_list = [len(sample) for sample in batch_sample]
        max_step = max_step if max_step else np.max(step_list)
        num_filters = batch_sample[0].shape[1]

        align_output = np.zeros((batch_size, max_step, num_filters)).astype(np.float32)
        for idx, sample in enumerate(batch_sample):
            align_output[idx, :len(sample)] = np.asarray(sample)
    
        return align_output, step_list


if __name__ == "__main__":
    audio_processer = AudioPorcesser('../data/thchs30/', 26, 2, 3, '../output/')
    lexicon = audio_processer.prepare('data_thchs30')
    print(audio_processer.get_data(0, 2, 'test', 'MAP'))
    #batch_sample = []
    #batch_sample.append(padding_context(np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5, 5], [6, 6, 6]]), 2).astype('float32'))
    #batch_sample.append(padding_context(np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5, 5], [6, 6, 6], [7, 7, 7]]), 2).astype('float32'))
    #print(align_batch_sample(batch_sample))
