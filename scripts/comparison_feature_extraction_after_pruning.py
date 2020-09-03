import os
import sys
pardir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(pardir)
from util_func import parameter_use, pruningtestacc_vis, batch_elastic_transform, list_of_norms, result_save, parameter_save
from loss import *
from dataset import *
from model import class_num, in_channel_num
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import math
learning_history = {'epoch': [], 'train_acc': [], 'train_total_loss': [], 'train_class_loss': [], 'train_ae_loss': [],
                    'train_error_1_loss': [], 'train_error_2_loss': [], 'test_acc': []}
learning_history['epoch'].append(0)
learning_history['train_acc'].append(0)
learning_history['train_total_loss'].append(0)
learning_history['train_class_loss'].append(0)
learning_history['train_ae_loss'].append(0)
learning_history['train_error_1_loss'].append(0)
learning_history['train_error_2_loss'].append(0)
learning_history['test_acc'].append(0.9920)

prototype = "15"
# training parameters
num_epochs = 20
save_step = 10
# elastic deformation parameters
sigma = 4
alpha = 20

original_net = parameter_use(f'./result/pkl/prototype_{prototype}/train_model_epoch500_{prototype}.pkl')
original_dense_weight = original_net.classifier[0].weight
for k in range(2, 11):
    print(f'count_{k}')
    train_net = parameter_use(f'./result/pkl/prototype_{prototype}/prune_dense/not_abs/prune_conv_proto_finetune_from_small/'
                              f'prune_train_model_epoch{k}_{prototype}.pkl')
    optimizer = optim.Adam(train_net.parameters(), lr=0.002)
    with torch.no_grad():
        train_net.classifier[0].weight = original_dense_weight
        # train_net.classifier[0].weight.data = torch.tensor(torch.rand(class_num, int(prototype)), device=device, dtype=dtype)

    for param in train_net.parameters():
        param.requires_grad = False
    train_net.prototype_feature_vectors.requires_grad = True
    for dense in train_net.classifier:
        dense.weight.requires_grad = True
    tmp_grad = torch.zeros(train_net.prototype_feature_vectors.shape).clone().detach().to(device)
    for epoch in range(num_epochs):
        # train
        train_net.train()
        train_loss, train_acc, train_class_error, train_ae_error, train_error_1, train_error_2 = 0, 0, 0, 0, 0, 0
        for i, (images, labels) in enumerate(train_loader):
            elastic_images = batch_elastic_transform(images.reshape(-1, in_height * in_width), sigma=sigma,
                                                     alpha=alpha, height=in_height, width=in_width) \
                .reshape(-1, in_channel_num, in_height, in_width)
            elastic_images, labels = torch.tensor(elastic_images, dtype=dtype).to(device), labels.to(device)
            optimizer.zero_grad()
            ae_output, prototype_distances, feature_vector_distances, outputs, softmax_output = train_net(
                elastic_images)
            class_error = criterion(outputs, labels)
            train_class_error += criterion(outputs, labels)
            ae_error = torch.mean(list_of_norms(ae_output - images.to(device)))
            train_ae_error += torch.mean(list_of_norms(ae_output - images.to(device)))
            error_1 = torch.mean(torch.min(feature_vector_distances, 1)[0])
            train_error_1 += torch.mean(torch.min(feature_vector_distances, 1)[0])
            error_2 = torch.mean(torch.min(prototype_distances, 1)[0])
            train_error_2 += torch.mean(torch.min(prototype_distances, 1)[0])
            loss = prototype_loss(class_error, ae_error, error_1, error_2, error_1_flag=True, error_2_flag=True)
            train_loss += loss.item()
            train_acc += (outputs.max(1)[1] == labels).sum().item()
            loss.backward()
            optimizer.step()
        avg_train_loss, avg_train_acc = train_loss / len(train_loader.dataset), train_acc / len(train_loader.dataset)
        avg_train_class_error, avg_train_ae_error = train_class_error / len(train_loader.dataset), train_ae_error / len(
            train_loader.dataset)
        avg_train_error_1, avg_train_error_2 = train_error_1 / len(train_loader.dataset), train_error_2 / len(
            train_loader.dataset)
        print(f'epoch [{epoch + 1}/{num_epochs}], train_loss: {avg_train_loss:.4f}, train_acc: {avg_train_acc:.4f}')

        # test
        avg_test_acc, test_acc = 0, 0
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            ae_output, prototype_distances, feature_vector_distances, outputs, softmax_output = train_net(
                images)
            test_acc += (outputs.max(1)[1] == labels).sum().item()
        avg_test_acc = test_acc / len(test_loader.dataset)
        print(f'test_acc: {avg_test_acc:.4f}')

        # save the learning history
        if epoch == num_epochs - 1:
            learning_history['epoch'].append(epoch + 1)
            learning_history['train_acc'].append(f'{avg_train_acc:.4f}')
            learning_history['train_total_loss'].append(f'{avg_train_loss:.4f}')
            learning_history['train_class_loss'].append(f'{avg_train_class_error:.4e}')
            learning_history['train_ae_loss'].append(f'{avg_train_ae_error:.4e}')
            learning_history['train_error_1_loss'].append(f'{avg_train_error_1:.4e}')
            learning_history['train_error_2_loss'].append(f'{avg_train_error_2:.4e}')
            learning_history['test_acc'].append(f'{avg_test_acc:.4f}')
            result_save(f'./result/csv/prune_train_history_{prototype}.csv', learning_history)

        # prototype
        if epoch == num_epochs - 1:
            with torch.no_grad():
                parameter_save(f'./result/pkl/prune_train_model_epoch{k}_{prototype}.pkl', train_net)

                f_width = int(math.sqrt(len(train_net.prototype_feature_vectors[1]) / class_num))
                f_height = int(math.sqrt(len(train_net.prototype_feature_vectors[1]) / class_num))
                prototype_imgs = train_net.decoder(
                    # prototype_imgs = net.cifar_decoder(
                    train_net.prototype_feature_vectors.reshape(int(prototype), class_num, f_width,
                                                                f_height)).cpu().numpy()
                n_cols = 5
                n_rows = int(prototype) // n_cols + 1 if int(prototype) % n_cols != 0 else int(prototype) // n_cols
                g, b = plt.subplots(n_rows, n_cols, figsize=(n_cols, n_rows), squeeze=False)
                for i in range(n_rows):
                    for j in range(n_cols):
                        if i * n_cols + j < int(prototype):
                            b[i][j].imshow(
                                prototype_imgs[i * n_cols + j].reshape(in_height, in_width),
                                # prototype_imgs[i * n_cols + j].reshape(in_height, in_width, in_channel_num),
                                cmap='gray',
                                interpolation='none')
                            b[i][j].axis('off')
                plt.savefig(f'./result/png/prune_prototype_epoch{k}_{prototype}.png',
                            transparent=True, bbox_inches='tight', pad_inches=0)
                # plt.show()
                plt.close()

pruningtestacc_vis(f'./result/png/prototype_{prototype}/testacc_afterprune.png', class_num, learning_history)

