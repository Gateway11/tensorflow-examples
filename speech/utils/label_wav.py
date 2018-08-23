import os
import numpy as np
import scipy.io.wavfile as wav

from collections import Counter
from python_speech_features import mfcc

def load_wav_file(wav_path):
    wav_files = []
    for (dirpath, dirnames, filenames) in os.walk(wav_path):
        for filename in filenames:
            if filename.endswith(".wav") or filename.endswith(".WAV"):
                filename_path = os.sep.join([dirpath, filename])
                if os.stat(filename_path).st_size < 240000:
                    continue
                wav_files.append(filename_path)

    return wav_files

def load_label_file(label_file):
    labels_dict = {}
    with open(label_file, "r", encoding='utf-8') as f:
        for label in f:
            label = label.strip("\n")
            label_id, label_text = label.split(' ', 1)
            labels_dict[label_id] = label_text

    return labels_dict

def prepare_label_list(wav_files, labels_dict):
    labels = []
    new_wav_files = []
    for wav_file in wav_files:
        wav_id = os.path.basename(wav_file).split(".")[0]
        if wav_id in labels_dict:
            labels.append(labels_dict[wav_id])
            new_wav_files.append(wav_file)

    all_words = []
    for label in labels:
        all_words += [word for word in label]

    counter = Counter(all_words)
    count_pairs = sorted(counter.items(), key = lambda x: -x[1])

    words, _ = zip(*count_pairs)
    lexicon = dict(zip(words, range(len(words))))

    return lexicon, labels, new_wav_files

def preapre_wav_list(wav_files, num_input, path):
    sample_files = []
    #os.mkdir(path)
    for wav_file in wav_files:
        fs, audio = wav.read(wav_file)
        orig_inputs = mfcc(audio, samplerate = fs, numcep = num_input)

        file_name = path + os.path.basename(wav_file).split(".")[0] + ".txt"
        np.savetxt(file_name, orig_inputs)
        sample_files.append(file_name);

    np.savetxt(path + ".complete", [len(sample_files)])
    return sample_files

def labels2vec(labels, lexicon):
    to_num = lambda word: lexicon.get(word, len(lexicon))
    return [list(map(to_num, label)) for label in labels]

def sparse_tuple_from(sequences, dtype = np.int32):
    indices = []
    values = []

    for n, seq in enumerate(sequences):
        indices.extend(zip([n] * len(seq), range(len(seq))))
        values.extend(seq)

    indices = np.asarray(indices, dtype = np.int64)
    values = np.asarray(values, dtype = dtype)
    shape = np.asarray([len(sequences), indices.max(0)[1] + 1], dtype = np.int64)

    return indices, values, shape

if __name__ == "__main__":
    wav_files = load_wav_file("/Users/daixiang/deep-learning/tensorflow/data/data_wsj/wav/train")
    labels_dict = load_label_file("/Users/daixiang/deep-learning/tensorflow/data/data_wsj/doc/trans/train.word.txt")

    lexicon, labels, wav_files = prepare_label_list(wav_files, labels_dict)
    vector_labels = labels2vec(labels, lexicon)

    sample = 1027
    print(wav_files[sample])
    print(labels[sample])
    print(vector_labels[sample])
    print(list(lexicon.keys())[6])

    sparse_labels = sparse_tuple_from(vector_labels[:8])
    #print(sparse_labels)

    #sample_files = preapre_wav_list(wav_files, 26, "./exp/")
