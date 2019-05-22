import torch
import torch.nn as nn

import numpy as np

import pathlib

import gensim

from tqdm import tqdm

from scipy.spatial.distance import cosine

CIFAR_IX_TO_CLASS = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat',
                     4: 'deer', 5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship',
                     9: 'truck'}
CIFAR_CLASS_TO_IX = {v:k for (k, v) in CIFAR_IX_TO_CLASS.items()}

def save_w2vec_classes(dataset='cifar10',
                       model_dir='/datasets_local/zsd_segmentation_pascalVOC/word_embedding/GoogleNews-vectors-negative300.bin.gz',
                       target_dir='/home/docker_user/local_storage/class_embedding'):
    """
    Save the w2vec embedding of classes in a given dataset into np.array

    Input:
        - dataset: str, name of the dataset (ex: 'cifar10', 'imagenet', 'nmnist')
        - model_dir: str, path of the w2vec embedding of google model trained
        over wikipedia
        - target_dir: str, path of
    """
    model_dir = pathlib.Path(model_dir)
    target_dir = pathlib.Path(target_dir).joinpath(dataset)
    target_dir.mkdir(parents=True, exist_ok=True)
    g_model = model = gensim.models.KeyedVectors.load_word2vec_format(model_dir,
                                                                      binary=True)
    if dataset == 'cifar10':
        ix_to_class_map = CIFAR_IX_TO_CLASS
    for class_name in tqdm(ix_to_class_map.values()):
        file_path = target_dir.joinpath(f'{class_name}_w2vec')
        np.save(file_path, g_model[class_name])

    print(f'All w2vec representation of classes from {dataset} saved at {str(target_dir)}')


def get_soft_label(label, dataset='cifar10',
                   w2vec_dir='/home/docker_user/local_storage/class_embedding',
                   method='softmax'):
    """
    Returns the soft label using cosine on w2vec representation of classes with
    a softmax or a uniform normalization

    Input:
        - label: int, index of the gt class
        - dataset: str
        - method: str, 'softmax' or 'uniform'

    Output:
         - soft_labels: torch.FloatTensor of size (batch_size, number_classes)
    """
    w2vec_dir = pathlib.Path(w2vec_dir).joinpath(dataset)
    if dataset == 'cifar10':
        class_to_ix_map = CIFAR_IX_TO_CLASS
    tmp_w2vec_class = np.array([np.load(w2vec_dir.joinpath(f'{label}_w2vec.npy'))
                                for label in list(class_to_ix_map.values())])
    gt_w2vec = tmp_w2vec_class[label, :]
    soft_labels = np.zeros(len(class_to_ix_map))
    for i in range(soft_labels.size):
        soft_labels[i] = 1 - cosine(tmp_w2vec_class[i, :], gt_w2vec)
    if method == 'softmax':
        softmax = nn.Softmax(dim=0)
        soft_labels = softmax(torch.FloatTensor(soft_labels))
    elif method == 'uniform':
        soft_labels = torch.FloatTensor(soft_labels / soft_labels.sum())

    return soft_labels

def get_batch_soft_labels(batch_labels, dataset='cifar10', method='softmax'):
    """
    Returns the soft label of a batch sample from get_soft_label function

    Input:
        - batch_labels: torch.LongTensor of size (batch_size)
        - dataset: str
        - method: str, 'softmax' or 'uniform'

    Output:
        - batch_soft_labels: torch.FloatTensor of size (batch_size, nb_classes)
    """
    batch_soft_labels = []
    for l in batch_labels:
        label = int(l)
        batch_soft_labels.append(get_soft_label(label, dataset=dataset, method=method).unsqueeze(0))

    return torch.cat(batch_soft_labels)

