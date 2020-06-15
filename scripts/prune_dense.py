import os
import sys
pardir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(pardir)
from densemask_generator import DenseMaskGenerator
from util_func import parameter_use
from dataset import *
import model
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

learning_history = {'epoch': [], 'train_acc': [], 'train_total_loss': [], 'train_class_loss': [], 'train_ae_loss': [],
                    'train_error_1_loss': [], 'train_error_2_loss': [], 'test_acc': []}

prototype = "5"
train_net = parameter_use(f'./result/pkl/prototype_{prototype}/train_model_epoch500_{prototype}.pkl')
optimizer = optim.Adam(train_net.parameters(), lr=0.002)
weight_matrix = train_net.classifier[0].weight
de_mask = torch.ones(weight_matrix.shape)

# weight_pruning
for count in range(model.class_num-1):
    print(f'\nweight pruning: {count+1}')
    # pruning from the large positive value
    prune_index_list = weight_matrix.detach().cpu().numpy().argmax(axis=0)
    # pruning from the small absolute value
    # prune_index_list = abs(weight_matrix.detach().cpu().numpy()).argmin(axis=0)
    print(prune_index_list)
    for i, index in enumerate(prune_index_list):
        de_mask[index, i] = 0
    with torch.no_grad():
        weight_matrix *= torch.tensor(de_mask, device=device, dtype=dtype)
        # test
        avg_test_acc, test_acc = 0, 0
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            ae_output, prototype_distances, feature_vector_distances, outputs, softmax_output = train_net(images)
            test_acc += (outputs.max(1)[1] == labels).sum().item()
        avg_test_acc = test_acc / len(test_loader.dataset)
        print(weight_matrix)
        print(f'test_acc: {avg_test_acc:.4f}')
