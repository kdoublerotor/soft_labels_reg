import torch

import numpy as np

import pathlib

CIFAR_IX_TO_CLASS = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat',
                     4: 'deer', 5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship',
                     9: 'truck'}


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

