import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

device = 'cuda'
dtype = torch.float
criterion = nn.CrossEntropyLoss()
batch_size = 250
in_height, in_width = 28, 28


def split_train_val(dataset, train_rate):
    n_samples = len(dataset)
    train_size = int(len(dataset) * train_rate)
    val_size = n_samples - train_size
    return torch.utils.data.random_split(dataset, [train_size, val_size])


def batch_elastic_transform(images, sigma, alpha, height, width, random_state=None):
    """
    this code is borrowed from chsasank on GitHubGist
    Elastic deformation of images as described in [Simard 2003].

    images: a two-dimensional numpy array; we can think of it as a list of flattened images
    sigma: the real-valued variance of the gaussian kernel
    alpha: a real-value that is multiplied onto the displacement fields

    returns: an elastically distorted image of the same shape
    """
    # import dataset
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
    return e_images.reshape(-1, in_height*in_width)


transform_train = transforms.Compose([
    # transforms.Grayscale(num_output_channels=1),  # cifar10
    # transforms.RandomRotation((-180, 180)),
    transforms.ToTensor(),
    # transforms.Normalize((0.5,), (0.5,))  # cifar10
])
transform_test = transforms.Compose([
    # transforms.Grayscale(num_output_channels=1),  # cifar10
    transforms.ToTensor(),
    # transforms.Normalize((0.5,), (0.5,))  # cifar10
])

train_val_dataset = torchvision.datasets.MNIST(root='./data/',
# train_val_dataset = torchvision.datasets.FashionMNIST(root='./data/',
# train_val_dataset = torchvision.datasets.CIFAR10(root='./data/',
                                                 train=True,
                                                 transform=transform_train,
                                                 download=True)
train_dataset, val_dataset = split_train_val(train_val_dataset, 0.8)
# print(train_dataset.dataset)
# print(train_dataset.indices.shape)
# exit()
test_dataset = torchvision.datasets.MNIST(root='./data/',
# test_dataset = torchvision.datasets.FashionMNIST(root='./data/',
# test_dataset = torchvision.datasets.CIFAR10(root='./data/',
                                            train=False,
                                            transform=transform_test,
                                            download=True)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           num_workers=2)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                         batch_size=batch_size,
                                         shuffle=True,
                                         num_workers=2)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False,
                                          num_workers=2)

# check if the function batch_elastic_transform works
# if __name__ == "__main__":
#     import matplotlib.pyplot as plt
#
#     '''
#     the following code demonstrates how gaussian_filter works by ploting
#     the displacement field before and after applying the gaussian_filter
#     '''
#     # random_state = np.random.RandomState(None)
#     # dx1 = random_state.rand(28, 28) * 2 - 1
#     # dy1 = random_state.rand(28, 28) * 2 - 1
#     # dx2 = gaussian_filter(dx1, 4, mode='constant')
#     # dy2 = gaussian_filter(dy1, 4, mode='constant')
#     # x, y = np.mgrid[0:28, 0:28]
#     # plt.quiver(x, y, dx1, dy1)
#     # plt.show()
#     # plt.quiver(x, y, dx2, dy2)
#     # plt.show()
#
#     for images, labels in train_loader:
#         plt.imshow(images[0].reshape(28, -1), cmap='gray')
#         plt.show()
#         dimg = batch_elastic_transform(images.reshape(-1, 784), sigma=4, alpha=20, height=28, width=28)
#         dimg = dimg.reshape(-1, 28, 28)
#         plt.imshow(dimg[0], cmap='gray')
#         plt.show()
#         plt.close()
#         break
