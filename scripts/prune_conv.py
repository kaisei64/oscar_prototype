import os
import sys
pardir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(pardir)
from channel_mask_generator import ChannelMaskGenerator
from util_func import parameter_use, testacc_vis, result_save, batch_elastic_transform, list_of_norms, result_save, parameter_save
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

prototype = "10"
train_net = parameter_use(f'./result/pkl/prototype_{prototype}/train_model_epoch500_{prototype}.pkl')
optimizer = optim.SGD(train_net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
# training parameters
num_epochs = 10
save_step = 10
# elastic deformation parameters
sigma = 4
alpha = 20

conv_list = [module for module in train_net.modules() if isinstance(module, nn.Conv2d)]
conv_count = len(conv_list)
ch_mask = [ChannelMaskGenerator() for _ in range(conv_count)]

inv_prune_ratio = 10
# channel_pruning
for count in range(1, inv_prune_ratio):
    print(f'\nchannel pruning: {count}')
    # keep norm sum
    channel_l1norm_for_each_layer = [list() for _ in range(conv_count)]
    # get norm
    for i, conv in enumerate(conv_list):
        channel_l1norm_for_each_layer[i] = [np.sum(torch.abs(param).cpu().detach().numpy()) for param in conv.weight]
        # sort in ascending order-----------------------------------
        channel_l1norm_for_each_layer[i].sort()
        # sort in descending order----------------------------------
        # channel_l1norm_for_each_layer[i].sort(reverse=True)

    with torch.no_grad():
        for i in range(len(conv_list)):
            if i != 0:
                ch_mask[i].generate_mask(conv_list[i].weight.data.clone(),
                                         None if i == 0 else conv_list[i - 1].weight.data.clone(), 0)
                continue
            threshold = channel_l1norm_for_each_layer[i][int(conv_list[i].out_channels / inv_prune_ratio * count)]\
                if count <= 9 else channel_l1norm_for_each_layer[i][int(conv_list[i].out_channels *
                                                        (9 / inv_prune_ratio + (count - 9) / inv_prune_ratio ** 2))]
            save_mask = ch_mask[i].generate_mask(conv_list[i].weight.data.clone(),
                                                 None if i == 0 else conv_list[i - 1].weight.data.clone(), threshold)
            conv_list[i].weight.data *= torch.tensor(save_mask, device=device, dtype=dtype)

    # parameter ratio
    weight_ratio = [np.count_nonzero(conv.weight.cpu().detach().numpy()) / np.size(conv.weight.cpu().detach().numpy())
                    for conv in conv_list]

    # number of channels after pruning
    channel_num_new = [conv.out_channels - ch_mask[i].channel_number(conv.weight) for i, conv in enumerate(conv_list)]
    for i in range(conv_count):
        print(f'conv{i + 1}_param: {weight_ratio[i]:.4f}', end=", " if i != conv_count - 1 else "\n")
    for i in range(conv_count):
        print(f'channel_number{i + 1}: {channel_num_new[i]}', end=", " if i != conv_count - 1 else "\n")

    # weight_pruning_not_finetune, only test-----------------------------------------------------------------------
#     with torch.no_grad():
#         avg_test_acc, test_acc = 0, 0
#         for images, labels in test_loader:
#             images, labels = images.to(device), labels.to(device)
#             ae_output, prototype_distances, feature_vector_distances, outputs, softmax_output = train_net(images)
#             test_acc += (outputs.max(1)[1] == labels).sum().item()
#         avg_test_acc = test_acc / len(test_loader.dataset)
#         print(f'test_acc: {avg_test_acc:.4f}')
#         learning_history['test_acc'].append(avg_test_acc)
#         parameter_save(f'./result/pkl/conv_prune_nofinetune_train_model_prune{count}_{prototype}.pkl', train_net)
#         # prototype
#         f_width = int(math.sqrt(len(train_net.prototype_feature_vectors[1]) / class_num))
#         f_height = int(math.sqrt(len(train_net.prototype_feature_vectors[1]) / class_num))
#         prototype_imgs = train_net.decoder(
#             # prototype_imgs = net.cifar_decoder(
#             train_net.prototype_feature_vectors.reshape(int(prototype), class_num, f_width, f_height)).cpu().numpy()
#         n_cols = 5
#         n_rows = int(prototype) // n_cols + 1 if int(prototype) % n_cols != 0 else int(prototype) // n_cols
#         g, b = plt.subplots(n_rows, n_cols, figsize=(n_cols, n_rows), squeeze=False)
#         for i in range(n_rows):
#             for j in range(n_cols):
#                 if i * n_cols + j < int(prototype):
#                     b[i][j].imshow(
#                         prototype_imgs[i * n_cols + j].reshape(in_height, in_width),
#                         # prototype_imgs[i * n_cols + j].reshape(in_height, in_width, in_channel_num),
#                         cmap='gray',
#                         interpolation='none')
#                     b[i][j].axis('off')
#         plt.savefig(f'./result/png/conv_prune_nofinetune_prototype_prune{count}_{prototype}.png',
#                     transparent=True, bbox_inches='tight', pad_inches=0)
#         # plt.show()
#         plt.close()
# testacc_vis(f'./result/png/conv_prune_nofinetune_testacc_afterprune.png', class_num - 1, learning_history)

    # weight_pruning_finetune----------------------------------------------------------------------------------------
    for param in train_net.parameters():
        param.requires_grad = False
    # train_net.prototype_feature_vectors.requires_grad = True
    for conv in conv_list:
        conv.weight.requires_grad = True
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
            ae_output, prototype_distances, feature_vector_distances, outputs, softmax_output = train_net(elastic_images)
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
            with torch.no_grad():
                for j, conv in enumerate(conv_list):
                    conv.weight.data *= torch.tensor(ch_mask[j].mask, device=device, dtype=dtype)
        avg_train_loss, avg_train_acc = train_loss / len(train_loader.dataset), train_acc / len(train_loader.dataset)
        avg_train_class_error, avg_train_ae_error = train_class_error / len(train_loader.dataset), train_ae_error / len(train_loader.dataset)
        avg_train_error_1, avg_train_error_2 = train_error_1 / len(train_loader.dataset), train_error_2 / len(train_loader.dataset)
        print(f'epoch [{epoch + 1}/{num_epochs}], train_loss: {avg_train_loss:.4f}, train_acc: {avg_train_acc:.4f}')

        # test
        avg_test_acc, test_acc = 0, 0
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            ae_output, prototype_distances, feature_vector_distances, outputs, softmax_output = train_net(images)
            test_acc += (outputs.max(1)[1] == labels).sum().item()
        avg_test_acc = test_acc / len(test_loader.dataset)
        print(f'test_acc: {avg_test_acc:.4f}')

        # save the learning history
        if epoch == num_epochs-1:
            learning_history['epoch'].append(epoch + 1)
            learning_history['train_acc'].append(f'{avg_train_acc:.4f}')
            learning_history['train_total_loss'].append(f'{avg_train_loss:.4f}')
            learning_history['train_class_loss'].append(f'{avg_train_class_error:.4e}')
            learning_history['train_ae_loss'].append(f'{avg_train_ae_error:.4e}')
            learning_history['train_error_1_loss'].append(f'{avg_train_error_1:.4e}')
            learning_history['train_error_2_loss'].append(f'{avg_train_error_2:.4e}')
            learning_history['test_acc'].append(f'{avg_test_acc:.4f}')
            result_save(f'./result/csv/conv_prune_finetune_train_history_{prototype}.csv', learning_history)

        # prototype
        if epoch == num_epochs - 1:
            with torch.no_grad():
                parameter_save(f'./result/pkl/conv_prune_finetune_train_model_prune{count}_{prototype}.pkl', train_net)
                f_width = int(math.sqrt(len(train_net.prototype_feature_vectors[1]) / class_num))
                f_height = int(math.sqrt(len(train_net.prototype_feature_vectors[1]) / class_num))
                prototype_imgs = train_net.decoder(
                    # prototype_imgs = net.cifar_decoder(
                    train_net.prototype_feature_vectors.reshape(int(prototype), class_num, f_width, f_height)).cpu().numpy()
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
                plt.savefig(f'./result/png/conv_prune_finetune_prototype_prune{count}_{prototype}.png',
                            transparent=True, bbox_inches='tight', pad_inches=0)
                # plt.show()
                plt.close()

testacc_vis(f'./result/png/prototype_{prototype}/conv_prune_finetune_testacc_afterprune.png', class_num-1, learning_history)
