import os
import scipy.io.wavfile as wav

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


def get_next_batches(wav_files, labels, next_idx, batch_size):
    batches_wavs = []
    batches_labels = []
    for i in range(batch_size):
        pass

    return batches_wavs, batches_labels, next_idx

if __name__ == "__main__":
    wav_files = load_wav_file("/Users/daixiang/deep-learning/data/data_wsj/wav/train")
    labels_dict = load_label_file("/Users/daixiang/deep-learning/data/data_wsj/doc/trans/train.word.txt")
