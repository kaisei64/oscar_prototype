import os
import sys
pardir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(pardir)
from densemask_generator import DenseMaskGenerator
from util_func import parameter_use, pruningtestacc_vis, result_save, batch_elastic_transform, list_of_norms, result_save, parameter_save, conv_vis
from loss import *
from dataset import *
from model import class_num, in_channel_num
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.spatial.distance import euclidean, minkowski, chebyshev, cityblock

learning_history = {'epoch': [], 'train_acc': [], 'train_total_loss': [], 'train_class_loss': [], 'train_ae_loss': [],
                    'train_error_1_loss': [], 'train_error_2_loss': [], 'test_acc': []}

prototype = "15"
train_net = parameter_use(f'./result/pkl/prototype_{prototype}/train_model_epoch500_{prototype}.pkl')
# train_net = parameter_use(f'./result/pkl/prototype_{prototype}/prune_dense/not_abs/prune_proto_dense_finetune_from_small/'
#                           f'prune_train_model_epoch7_{prototype}.pkl')
optimizer = optim.Adam(train_net.parameters(), lr=0.002)
# training parameters
num_epochs = 5
save_step = 10
# elastic deformation parameters
sigma = 4
alpha = 20

weight_matrix = train_net.classifier[0].weight
de_mask = torch.ones(weight_matrix.shape)
proto_matrix = train_net.prototype_feature_vectors
proto_mask = torch.ones(proto_matrix.shape)
prune_proto_list = list()
# pruning from the large absolute value-----------------------------------------
prune_index_list = np.argsort(np.sum(abs(weight_matrix.detach().cpu().numpy()), axis=0))[::-1]
# pruning from the small absolute value-----------------------------------------
# prune_index_list = np.argsort(np.sum(abs(weight_matrix.detach().cpu().numpy()), axis=0))

# pruning only one prototype-------------------------------------------------------
# de_mask[:, 0] = 0
# proto_mask[0] = 0

for count in range(int(prototype)):
    print(f'\nprototype pruning: {count}')
    # if count != 0:
    #     de_mask[:, prune_index_list[count-1]] = 0
    #     proto_mask[prune_index_list[count-1]] = 0
    #     with torch.no_grad():
    #         weight_matrix *= torch.tensor(de_mask, device=device, dtype=dtype)
    #         proto_matrix *= torch.tensor(proto_mask, device=device, dtype=dtype)
    # weight_pruning_not_finetune, only test-----------------------------
    # with torch.no_grad():
    #     avg_test_acc, test_acc = 0, 0
    #     for images, labels in test_loader:
    #         images, labels = images.to(device), labels.to(device)
    #         ae_output, prototype_distances, feature_vector_distances, proto_proto_distances, outputs, softmax_output = train_net(images)
    #         test_acc += (outputs.max(1)[1] == labels).sum().item()
    #     avg_test_acc = test_acc / len(test_loader.dataset)
    #     print(f'test_acc: {avg_test_acc:.4f}')
    #     learning_history['test_acc'].append(avg_test_acc)

    # weight_pruning_finetune--------------------------------------------
    if count == 0:
        # test
        avg_test_acc, test_acc = 0, 0
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            ae_output, prototype_distances, feature_vector_distances, proto_proto_distances, outputs, softmax_output = train_net(images)
            test_acc += (outputs.max(1)[1] == labels).sum().item()
        avg_test_acc = test_acc / len(test_loader.dataset)
        print(f'test_acc: {avg_test_acc:.4f}')
        learning_history['epoch'].append(0)
        learning_history['train_acc'].append(0)
        learning_history['train_total_loss'].append(0)
        learning_history['train_class_loss'].append(0)
        learning_history['train_ae_loss'].append(0)
        learning_history['train_error_1_loss'].append(0)
        learning_history['train_error_2_loss'].append(0)
        learning_history['test_acc'].append(avg_test_acc)
        continue

    # prune the prototypes that are close to other prototypes----------------------------------------------------
    with torch.no_grad():
        proto_distance_list = list()
        for i in range(int(prototype)):
            tmp_distance_sum = 0
            for j in range(int(prototype)):
                if j not in prune_proto_list:
                    tmp_distance_sum += euclidean(train_net.prototype_feature_vectors[i].cpu().detach().numpy(),
                                                  train_net.prototype_feature_vectors[j].cpu().detach().numpy())
            # Exclude prototypes that have already been pruned
            if i in prune_proto_list:
                tmp_distance_sum = 10000000
            proto_distance_list.append(tmp_distance_sum)
        # tmp_prune_proto_list = np.argsort(np.array(proto_distance_list))[0] # near
        tmp_prune_proto_list = np.argsort(np.array(proto_distance_list))[0][::-1] # far
        # Add already pruned prototype
        prune_proto_list.append(tmp_prune_proto_list)
        print(f'\n{tmp_prune_proto_list + 1}th prototype pruning')
        de_mask[:, tmp_prune_proto_list] = 0
        proto_mask[tmp_prune_proto_list] = 0
        weight_matrix *= torch.tensor(de_mask, device=device, dtype=dtype)
        proto_matrix *= torch.tensor(proto_mask, device=device, dtype=dtype)

    for param in train_net.parameters():
        param.requires_grad = True
    # train_net.prototype_feature_vectors.requires_grad = True
    # for dense in train_net.classifier:
    #     dense.weight.requires_grad = True
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
            ae_output, prototype_distances, feature_vector_distances, proto_proto_distances, outputs, softmax_output = train_net(elastic_images)
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
                weight_matrix *= torch.tensor(de_mask, device=device, dtype=dtype)
                proto_matrix *= torch.tensor(proto_mask, device=device, dtype=dtype)
        avg_train_loss, avg_train_acc = train_loss / len(train_loader.dataset), train_acc / len(train_loader.dataset)
        avg_train_class_error, avg_train_ae_error = train_class_error / len(train_loader.dataset), train_ae_error / len(train_loader.dataset)
        avg_train_error_1, avg_train_error_2 = train_error_1 / len(train_loader.dataset), train_error_2 / len(train_loader.dataset)
        print(f'epoch [{epoch + 1}/{num_epochs}], train_loss: {avg_train_loss:.4f}, train_acc: {avg_train_acc:.4f}')

        # test
        avg_test_acc, test_acc = 0, 0
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            ae_output, prototype_distances, feature_vector_distances, proto_proto_distances, outputs, softmax_output = train_net(images)
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
                parameter_save(f'./result/pkl/prune_train_model_epoch{count + 1}_{prototype}.pkl', train_net)

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
                plt.savefig(f'./result/png/prune_prototype_epoch{count + 1}_{prototype}.png',
                            transparent=True, bbox_inches='tight', pad_inches=0)
                # plt.show()
                plt.close()

pruningtestacc_vis(f'./result/png/prototype_{prototype}/testacc_afterprune.png', int(prototype), learning_history)
