import numpy as np


def get_next_batches(next_idx, files, labels, num_contexts, batch_size):
    batches_label = []
    batches_sample = []
    for i in range(batch_size):
        sample = np.loadtxt(files[next_idx])
        sample = organization_context(sample[::2], num_contexts)
        batches_sample.append(sample)
        next_idx++

    return batches_labels, batches_samples, next_idx

def organization_context(sample, num_contexts):
    pass

def align_samples(batches_labels, batches_samples):
    pass

if __name__ == "__main__":
    pass
