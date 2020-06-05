import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from util_func import split_train_val, batch_elastic_transform

device = 'cuda'
dtype = torch.float
criterion = nn.CrossEntropyLoss()
batch_size = 250
in_height, in_width = 28, 28

transform_train = transforms.Compose([
    # transforms.Grayscale(num_output_channels=1),  # cifar10
    transforms.ToTensor(),
    # transforms.Normalize((0.5,), (0.5,))  # cifar10
])
transform_test = transforms.Compose([
    # transforms.Grayscale(num_output_channels=1),  # cifar10
    transforms.ToTensor(),
    # transforms.Normalize((0.5,), (0.5,))  # cifar10
])

# train_val_dataset = torchvision.datasets.MNIST(root='./data/',
train_val_dataset = torchvision.datasets.FashionMNIST(root='./data/',
# train_val_dataset = torchvision.datasets.CIFAR10(root='./data/',
                                                 train=True,
                                                 transform=transform_train,
                                                 download=True)
train_dataset, val_dataset = split_train_val(train_val_dataset, 0.8)
# test_dataset = torchvision.datasets.MNIST(root='./data/',
test_dataset = torchvision.datasets.FashionMNIST(root='./data/',
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
