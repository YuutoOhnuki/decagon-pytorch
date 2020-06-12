import tensorflow as tf
import numpy as np
from collections import defaultdict
import torch
import torch.utils.data
from typing import List, \
    Union
import decagon_pytorch.sampling
import scipy.stats


def test_unigram_01():
    range_max = 7
    distortion = 0.75
    batch_size = 500
    unigrams = [ 1, 3, 2, 1, 2, 1, 3]
    num_true = 1

    true_classes = np.zeros((batch_size, num_true), dtype=np.int64)
    for i in range(batch_size):
        true_classes[i, 0] = i % range_max
    true_classes = tf.convert_to_tensor(true_classes)

    neg_samples, _, _ = tf.nn.fixed_unigram_candidate_sampler(
        true_classes=true_classes,
        num_true=num_true,
        num_sampled=batch_size,
        unique=False,
        range_max=range_max,
        distortion=distortion,
        unigrams=unigrams)

    assert neg_samples.shape == (batch_size,)

    for i in range(batch_size):
        assert neg_samples[i] != true_classes[i, 0]

    counts = defaultdict(int)
    with tf.Session() as sess:
        neg_samples = neg_samples.eval()
    for x in neg_samples:
        counts[x] += 1

    print('counts:', counts)

    assert counts[0] < counts[1] and \
        counts[0] < counts[2] and \
        counts[0] < counts[4] and \
        counts[0] < counts[6]

    assert counts[2] < counts[1] and \
        counts[0] < counts[6]

    assert counts[3] < counts[1] and \
        counts[3] < counts[2] and \
        counts[3] < counts[4] and \
        counts[3] < counts[6]

    assert counts[4] < counts[1] and \
        counts[4] < counts[6]

    assert counts[5] < counts[1] and \
        counts[5] < counts[2] and \
        counts[5] < counts[4] and \
        counts[5] < counts[6]


def test_unigram_02():
    range_max = 7
    distortion = 0.75
    batch_size = 500
    unigrams = [ 1, 3, 2, 1, 2, 1, 3]
    num_true = 1

    true_classes = np.zeros((batch_size, num_true), dtype=np.int64)
    for i in range(batch_size):
        true_classes[i, 0] = i % range_max
    true_classes = torch.tensor(true_classes)

    neg_samples = decagon_pytorch.sampling.fixed_unigram_candidate_sampler(
        true_classes=true_classes,
        num_samples=batch_size,
        distortion=distortion,
        unigrams=unigrams)

    assert neg_samples.shape == (batch_size,)

    for i in range(batch_size):
        assert neg_samples[i] != true_classes[i, 0]

    counts = defaultdict(int)
    for x in neg_samples:
        counts[x] += 1

    print('counts:', counts)

    assert counts[0] < counts[1] and \
        counts[0] < counts[2] and \
        counts[0] < counts[4] and \
        counts[0] < counts[6]

    assert counts[2] < counts[1] and \
        counts[0] < counts[6]

    assert counts[3] < counts[1] and \
        counts[3] < counts[2] and \
        counts[3] < counts[4] and \
        counts[3] < counts[6]

    assert counts[4] < counts[1] and \
        counts[4] < counts[6]

    assert counts[5] < counts[1] and \
        counts[5] < counts[2] and \
        counts[5] < counts[4] and \
        counts[5] < counts[6]


def test_unigram_03():
    range_max = 7
    distortion = 0.75
    batch_size = 25
    unigrams = [ 1, 3, 2, 1, 2, 1, 3]
    num_true = 1

    true_classes = np.zeros((batch_size, num_true), dtype=np.int64)
    for i in range(batch_size):
        true_classes[i, 0] = i % range_max

    true_classes_tf = tf.convert_to_tensor(true_classes)
    true_classes_torch = torch.tensor(true_classes)

    counts_tf = defaultdict(list)
    counts_torch = defaultdict(list)

    for i in range(100):
        neg_samples, _, _ = tf.nn.fixed_unigram_candidate_sampler(
            true_classes=true_classes_tf,
            num_true=num_true,
            num_sampled=batch_size,
            unique=False,
            range_max=range_max,
            distortion=distortion,
            unigrams=unigrams)

        counts = defaultdict(int)
        with tf.Session() as sess:
            neg_samples = neg_samples.eval()
        for x in neg_samples:
            counts[x] += 1
        for k, v in counts.items():
            counts_tf[k].append(v)

        neg_samples = decagon_pytorch.sampling.fixed_unigram_candidate_sampler(
            true_classes=true_classes,
            num_samples=batch_size,
            distortion=distortion,
            unigrams=unigrams)

        counts = defaultdict(int)
        for x in neg_samples:
            counts[x] += 1
        for k, v in counts.items():
            counts_torch[k].append(v)

    for i in range(range_max):
        print('counts_tf[%d]:' % i, counts_tf[i])
        print('counts_torch[%d]:' % i, counts_torch[i])

    for i in range(range_max):
        statistic, pvalue = scipy.stats.ttest_ind(counts_tf[i], counts_torch[i])
        assert pvalue * range_max > .05
