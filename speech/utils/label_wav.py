import os
import numpy as np
import scipy.io.wavfile as wav

from collections import Counter
from python_speech_features import mfcc


def list_wav_file(wav_path):
    wav_files = []
    for (dirpath, dirnames, filenames) in os.walk(wav_path):
        for filename in filenames:
            if filename.endswith('.wav') or filename.endswith('.WAV'):
                filename_path = os.sep.join([dirpath, filename])
                if os.stat(filename_path).st_size < 240000:
                    continue
                wav_files.append(filename_path)

    return wav_files


def load_label_file(filename):
    labels_dict = {}
    with open(filename, 'r', encoding='utf-8') as f:
        for label in f:
            label = label.rstrip()
            label_id, label_content = label.split(' ', 1)
            labels_dict[label_id] = label_content 

    return labels_dict


def prepare_label_list(sample_files, labels_dict):
    labels = []
    new_wav_files = []
    for sample_file in sample_files:
        wav_id = os.path.basename(sample_file).split('.')[0]
        if wav_id in labels_dict:
            labels.append(labels_dict[wav_id])
            new_wav_files.append(sample_file)

    all_symols = []
    for label in labels:
        all_symols += label.split(' ')

    counter = Counter(all_symols)
    count_pairs = sorted(counter.items())

    symbols, _ = zip(*count_pairs)
    lexicon = dict(zip(symbols, range(len(symbols))))
    lexicon['<unk>'] = len(lexicon)

    return lexicon, labels, new_wav_files


def preapre_wav_list(wav_files, num_inputs, downsampling_ratio, output_path):
    complete_filename = output_path + '.complete'

    print('mfcc output dir:', output_path)
    if os.path.exists(complete_filename):
        max_step = np.loadtxt(complete_filename, 'int32')
        return [output_path + file_name for file_name \
                in os.listdir(output_path) if file_name.endswith('.txt')], max_step

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    max_step = 0
    sample_files = []
    for wav_file in wav_files:
        fs, audio = wav.read(wav_file)
        orig_inputs = mfcc(audio, samplerate=fs, numcep=num_inputs)[::downsampling_ratio]
        max_step = max_step if not len(orig_inputs) > max_step else len(orig_inputs)

        file_name = output_path + os.path.basename(wav_file).split(".")[0] + ".txt"
        np.savetxt(file_name, orig_inputs)
        sample_files.append(file_name)

    np.savetxt(complete_filename, [max_step])
    return sample_files, max_step


def labels_to_vector(labels, lexicon):
    def to_num(symbol): return lexicon.get(symbol, lexicon['<unk>'])
    return [list(map(to_num, label.split(' '))) for label in labels]


def sparse_tuple_from(sequences, dtype=np.int32):
    """密集矩阵转稀疏矩阵
    """
    indices = []
    values = []

    for n, seq in enumerate(sequences):
        indices.extend(zip([n] * len(seq), range(len(seq))))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), indices.max(0)[1] + 1], dtype=np.int64)

    return indices, values, shape


def sparse_tuple_to_string(tuple, lexicon):
    """向量转换成文字
    """
    indices = tuple[0]
    values = tuple[1]
    results = [''] * tuple[2][0]

    words = list(lexicon.keys())
    for i in range(len(indices)):
        results[indices[i][0]] += ' ' if indices[i, 1] else ''
        results[indices[i][0]] += words[values[i]]

    return results

def vector_to_string(vec, lexicon):
    """向量转换成文字
    """

    vec_size = len(vec)
    results = ['']

    words = list(lexicon.keys())
    for i in range(vec_size):
        results[0] += ' ' if i else ''
        results[0] += words[vec[i]]

    return results 


def trans_array_to_string(value, lexicon):
    results = ''
    words = list(lexicon.keys())
    for i in range(len(value)):
        results += words[value[i]]  # chr(value[i] + FIRST_INDEX)
    return results.replace('`', ' ')


if __name__ == "__main__":
    wav_files = list_wav_file('../data/thchs30/data_thchs30/train')
    labels_dict = load_label_file('../data/thchs30/resource/trains/train.syllable.txt')

    lexicon, labels, wav_files = prepare_label_list(wav_files, labels_dict)
    labels_vector = labels_to_vector(labels, lexicon)
    import re
    with open('./symbol_table.txt', 'w') as f:
        f.write(re.sub('[\s\'{}]', '', str(lexicon)).replace(',', '\n').replace(':', '\t'))

    sample = 1027
    print(wav_files[sample])
    print(labels[sample])
    print(labels_vector[sample])
    print(list(lexicon.keys())[1041])

    sparse_labels = sparse_tuple_from(labels_vector[:3])
    decoded_str = trans_tuple_to_texts(sparse_labels, lexicon)
    # print(sparse_labels)
    print(decoded_str)

    sample_files, max_step = preapre_wav_list(wav_files, 26, 2, '../output/mfcc/data_thchs30/train/')
