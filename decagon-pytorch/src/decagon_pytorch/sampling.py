import numpy as np
import torch
import torch.utils.data
from typing import List, \
    Union


def fixed_unigram_candidate_sampler(
    true_classes: Union[np.array, torch.Tensor],
    num_samples: int,
    unigrams: List[Union[int, float]],
    distortion: float = 1.):

    if isinstance(true_classes, torch.Tensor):
        true_classes = true_classes.detach().cpu().numpy()
    if true_classes.shape[0] != num_samples:
        raise ValueError('true_classes must be a 2D matrix with shape (num_samples, num_true)')
    unigrams = np.array(unigrams)
    if distortion != 1.:
        unigrams = unigrams.astype(np.float64) ** distortion
    # print('unigrams:', unigrams)
    indices = np.arange(num_samples)
    result = np.zeros(num_samples, dtype=np.int64)
    while len(indices) > 0:
        # print('len(indices):', len(indices))
        sampler = torch.utils.data.WeightedRandomSampler(unigrams, len(indices))
        candidates = np.array(list(sampler))
        candidates = np.reshape(candidates, (len(indices), 1))
        # print('candidates:', candidates)
        # print('true_classes:', true_classes[indices, :])
        result[indices] = candidates.T
        mask = (candidates == true_classes[indices, :])
        mask = mask.sum(1).astype(np.bool)
        # print('mask:', mask)
        indices = indices[mask]
    return result
