import torch

import numpy as np

import pathlib

import gensim

from tqdm import tqdm

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
    model_dir = pathlib.Path(model_dir)
    target_dir = pathlib.Path(target_dir).joinpath(dataset)
    target_dir.mkdir(parents=True, exist_ok=True)
    g_model = model = gensim.models.KeyedVectors.load_word2vec_format(model_dir,
                                                                      binary=True)
    for class_name in tqdmlist(CIFAR_IX_TO_CLASS.values()):
        file_path = target_dir.joinpath(f'{class_name}_w2vec')
        np.save(file_path, g_model[class_name])

    print(f'All w2vec representation of classes from {dataset} saved at {str(target_dir)}')



