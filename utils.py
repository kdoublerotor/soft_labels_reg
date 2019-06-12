import torch
import torch.nn as nn

import numpy as np

import pathlib

import gensim

from tqdm import tqdm

import matplotlib.pyplot as plt

from scipy.spatial.distance import cosine

from sklearn.metrics import confusion_matrix


CIFAR_IX_TO_CLASS = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat',
                     4: 'deer', 5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship',
                     9: 'truck'}
CIFAR_CLASS_TO_IX = {v: k for (k, v) in CIFAR_IX_TO_CLASS.items()}


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
    target_dir = pathlib.Path(target_dir) / dataset
    target_dir.mkdir(parents=True, exist_ok=True)
    g_model = model = gensim.models.KeyedVectors.load_word2vec_format(model_dir,
                                                                      binary=True)
    if dataset == 'cifar10':
        ix_to_class_map = CIFAR_IX_TO_CLASS
    for class_name in tqdm(ix_to_class_map.values()):
        file_path = target_dir.joinpath(f'{class_name}_w2vec')
        np.save(file_path, g_model[class_name])

    print(f'All w2vec representation of classes from {dataset} saved at {str(target_dir)}')


def get_soft_label(label, dataset='cifar10', method='cosine', proba_reg='softmax',
                   alpha=2.0, w2vec_dir='/home/docker_user/local_storage/class_embedding'):
    """
    Returns the soft label using cosine / dot on w2vec representation of classes with
    a softmax or a uniform normalization

    Input:
        - label: int, index of the gt class
        - dataset: str
        - proba_reg: str, 'softmax' or 'power'
        - alpha: float, exponent of the 'power' proba_reg method
        - method: 'cosine' or 'dot' method to compute a metric in the class space

    Output:
         - soft_labels: torch.FloatTensor of size (number_classes)
    """
    w2vec_dir = pathlib.Path(w2vec_dir).joinpath(dataset)
    if dataset == 'cifar10':
        class_to_ix_map = CIFAR_IX_TO_CLASS
    tmp_w2vec_class = np.array([np.load(w2vec_dir.joinpath(f'{label}_w2vec.npy'))
                                for label in list(class_to_ix_map.values())])
    gt_w2vec = tmp_w2vec_class[label, :]
    soft_labels = np.zeros(len(class_to_ix_map))
    for i in range(soft_labels.size):
        if method == 'cosine':
            soft_labels[i] = cosine(tmp_w2vec_class[i, :], gt_w2vec)
            soft_labels = 1 - soft_labels
        elif method == 'dot':
            soft_labels[i] = np.dot(tmp_w2vec_class[i, :], gt_w2vec)
    if proba_reg == 'softmax':
        softmax = nn.Softmax(dim=0)
        soft_labels = softmax(torch.FloatTensor(soft_labels))
    elif proba_reg == 'power':
        soft_labels = torch.FloatTensor(
            soft_labels ** alpha / (soft_labels ** alpha).sum())

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
        batch_soft_labels.append(get_soft_label(
            label, dataset=dataset, method=method).unsqueeze(0))

    return torch.cat(batch_soft_labels)


def plot_distribution(soft_labels_dict, dataset='cifar10'):
    """
    Plots the distribution of a soft labl np.array

    Input:
       - soft_labels: np.array of szie (nb_classes)
       - dataset: str
    """
    if dataset == 'cifar10':
        class_to_ix_map = CIFAR_IX_TO_CLASS
    fig = plt.figure(figsize=(5, 30))
    pos = np.arange(len(soft_labels_dict[0]))
    max_prob = max([x.max() for x in soft_labels_dict.values()])
    for i in soft_labels_dict.keys():
        ax = plt.subplot(len(soft_labels_dict.keys()), 1, i + 1)
        ax.bar(pos, soft_labels_dict[i])
        ax.set_title(f'distribution of {CIFAR_IX_TO_CLASS[i]}')
        plt.xticks(pos, list(class_to_ix_map.values()), rotation=90)
        plt.ylim(0, max_prob)
    plt.tight_layout()
    plt.show()


def compute_confusion_matrix(model, data_loader, soft_labels=True):
    """

    """
    tqdm_batch = tqdm(data_loader, total=len(data_loader),
                      desc="Test run - ", ascii=True)

    model.eval()
    num_classes = len(data_loader.dataset.classes)
    cfm_labels = np.zeros((num_classes, num_classes), dtype=int)

    with torch.no_grad():
        for data, target in tqdm_batch:
            data = data.to('cuda', non_blocking=True)

            target1, target2 = target[0].to('cuda', non_blocking=True), \
                target[1].to('cuda', non_blocking=True)

            pred = model(data)

            threshold1 = 0.5
            m = nn.Softmax(dim=1)
            logits = m(pred.cpu())
            #labels_pred1 = logits1 > threshold1

            pred = logits.argmax(dim=1).numpy()

            if soft_labels == True:
                target = target.cpu().argmax(dim=1).numpy()

            cfm_labels += confusion_matrix(target,
                                           pred,
                                           labels=range(num_classes))

    tqdm_batch.close()

    return cfm_labels
