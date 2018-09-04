import os
import numpy as np
import scipy.io.wavfile as wav

from collections import Counter
from python_speech_features import mfcc
from xpinyin import Pinyin


pin = Pinyin()
def load_wav_file(wav_path):
    wav_files = []
    for (dirpath, dirnames, filenames) in os.walk(wav_path):
        for filename in filenames:
            if filename.endswith('.wav') or filename.endswith('.WAV'):
                filename_path = os.sep.join([dirpath, filename])
                if os.stat(filename_path).st_size < 240000:
                    continue
                wav_files.append(filename_path)

    return wav_files


def load_label_file(label_file):
    labels_dict = {}
    with open(label_file, 'r', encoding='utf-8') as f:
        for label in f:
            label = label.strip("\n")
            label_id, label_text = label.split(' ', 1)
            labels_dict[label_id] = label_text

    return labels_dict


def prepare_label_list(sample_files, labels_dict):
    labels = []
    new_wav_files = []
    for sample_file in sample_files:
        wav_id = os.path.basename(sample_file).split(".")[0]
        if wav_id in labels_dict:
            labels.append(labels_dict[wav_id])
            new_wav_files.append(sample_file)

    all_words = []
    for label in labels:
        all_words += [word for word in label]

    all_pinyins = pin.get_pinyin(''.join(all_words)).split('-')
    counter = Counter(all_pinyins)
    count_pairs = sorted(counter.items())

    pinyins, _ = zip(*count_pairs)
    lexicon = dict(zip(pinyins, range(len(pinyins))))

    return lexicon, labels, new_wav_files


def preapre_wav_list(wav_files, num_input, path):
    if os.path.exists(path + '.complete.txt'):
        return [path + file_name for file_name in os.listdir(path) if file_name.endswith('.txt')]

    if not os.path.exists(path):
        os.makedirs(path)

    sample_files = []
    for wav_file in wav_files:
        fs, audio = wav.read(wav_file)
        orig_inputs = mfcc(audio, samplerate=fs, numcep=num_input)

        file_name = path + os.path.basename(wav_file).split(".")[0] + ".txt"
        np.savetxt(file_name, orig_inputs[::2])
        sample_files.append(file_name)

    np.savetxt(path + '.complete.txt', [len(sample_files)])
    return sample_files


def labels_to_vector(labels, lexicon):
    def to_num(pinyin): return lexicon.get(pinyin, len(lexicon))
    return [list(map(to_num, pin.get_pinyin(''.join(label)).split('-'))) for label in labels]


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


def trans_tuple_to_texts(tuple, lexicon):
    """向量转换成文字
    """
    indices = tuple[0]
    values = tuple[1]
    results = [''] * tuple[2][0]

    words = list(lexicon.keys())
    for i in range(len(indices)):
        results[indices[i][0]] += words[values[i]]

    return results


def trans_array_to_text(value, lexicon):
    results = ''
    words = list(lexicon.keys())
    for i in range(len(value)):
        results += words[value[i]]  # chr(value[i] + FIRST_INDEX)
    return results.replace('`', ' ')


if __name__ == "__main__":
    wav_files = load_wav_file('/tmp/data_wsj/wav/train')
    labels_dict = load_label_file('/tmp/data_wsj/doc/trans/train.word.txt')

    pin = Pinyin()
    lexicon, labels, wav_files = prepare_label_list(wav_files, labels_dict)
    vector_labels = labels_to_vector(labels, lexicon)
    f = open('./symbol_table.txt', 'w')
    import re
    f.write(re.sub('[\s\'{}]', '', str(lexicon)).replace(',', '\n').replace(':', '\t'))
    f.close()

    sample = 1027
    print(wav_files[sample])
    print(labels[sample])
    print(vector_labels[sample])
    print(list(lexicon.keys())[11])

    sparse_labels = sparse_tuple_from(vector_labels[:3])
    decoded_str = trans_tuple_to_texts(sparse_labels, lexicon)
    # print(sparse_labels)
    print(decoded_str)

    sample_files = preapre_wav_list(wav_files, 26, '../output/mfcc/train/')
