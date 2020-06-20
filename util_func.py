import torch
import numpy as np
import cloudpickle
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter


def split_train_val(train_val_dataset, train_rate):
    n_samples = len(train_val_dataset)
    train_size = int(len(train_val_dataset) * train_rate)
    val_size = n_samples - train_size
    return torch.utils.data.random_split(train_val_dataset, [train_size, val_size])


def batch_elastic_transform(images, sigma, alpha, height, width, random_state=None):
    """
    this code is borrowed from chsasank on GitHubGist
    Elastic deformation of images as described in [Simard 2003].

    images: a two-dimensional numpy array; we can think of it as a list of flattened images
    sigma: the real-valued variance of the gaussian kernel
    alpha: a real-value that is multiplied onto the displacement fields

    returns: an elastically distorted image of the same shape
    """
    import dataset
    assert len(images.shape) == 2
    # the two lines below ensure we do not alter the array images
    e_images = np.empty_like(images)
    e_images[:] = images
    e_images = e_images.reshape(-1, height, width)

    if random_state is None:
        random_state = np.random.RandomState(None)

    x, y = np.mgrid[0:height, 0:width]
    for i in range(e_images.shape[0]):
        dx = gaussian_filter((random_state.rand(height, width) * 2 - 1), sigma, mode='constant') * alpha
        dy = gaussian_filter((random_state.rand(height, width) * 2 - 1), sigma, mode='constant') * alpha
        indices = x + dx, y + dy
        e_images[i] = map_coordinates(e_images[i], indices, order=1)
    return e_images.reshape(-1, dataset.in_height*dataset.in_width)


def list_of_distances(x, y):
    """
    Given a list of vectors, X = [x_1, ..., x_n], and another list of vectors,
    Y = [y_1, ... , y_m], we return a list of vectors
            [[d(x_1, y_1), d(x_1, y_2), ... , d(x_1, y_m)],
             ...
             [d(x_n, y_1), d(x_n, y_2), ... , d(x_n, y_m)]],
    where the distance metric used is the squared euclidean distance.
    The computation is achieved through a clever use of broadcasting.
    """
    xx = torch.reshape(list_of_norms(x), shape=(-1, 1))
    yy = torch.reshape(list_of_norms(y), shape=(1, -1))
    output = xx + yy - 2 * torch.matmul(x, torch.t(y))
    return output


def list_of_norms(x):
    """
    X is a list of vectors X = [x_1, ..., x_n], we return
        [d(x_1, x_1), d(x_2, x_2), ... , d(x_n, x_n)], where the distance
    function is the squared euclidean distance.
    """
    return torch.pow(x, 2).sum(dim=1)


def result_save(path, learning_history):
    df = pd.DataFrame.from_dict(learning_history)
    df.to_csv(path)


def parameter_save(path, param):
    with open(path, 'wb') as f:
        cloudpickle.dump(param, f)


def parameter_use(path):
    with open(path, 'rb') as f:
        return cloudpickle.load(f)


def loss_vis(path, epoch, history):
    plt.figure()
    plt.plot(range(1, epoch + 1), history['train_loss'], label='train_loss')
    plt.plot(range(1, epoch + 1), history['train_loss'], label='train_loss')
    plt.xlabel('epoch')
    plt.legend()
    plt.show()
    # plt.savefig(path)


def testacc_vis(path, epoch, history):
    plt.figure()
    history['test_acc'] = [float(val) for val in history['test_acc']]
    plt.plot(range(1, epoch + 1), history['test_acc'])
    plt.title('test accuracy')
    plt.xlabel('epoch')
    # plt.show()
    plt.savefig(path)
