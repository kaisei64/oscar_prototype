import os
import sys
pardir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(pardir)
from dataset import *
from model import ProtoNet, class_num, prototype_num, in_channel_num
from util_func import batch_elastic_transform, list_of_norms, result_save, parameter_save, list_of_distances, testacc_vis, parameter_use
from loss import *
import torch
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math
from scipy.spatial.distance import euclidean, minkowski, chebyshev, cityblock

learning_history = {'epoch': [], 'train_acc': [], 'train_total_loss': [], 'train_class_loss': [], 'train_ae_loss': [],
                    'train_error_1_loss': [], 'train_error_2_loss': [], 'val_acc': [], 'val_total_loss': [],
                    'val_class_loss': [], 'val_ae_loss': [], 'val_error_1_loss': [], 'val_error_2_loss': [],
                    'test_acc': []}

net = ProtoNet().to(device)
# net = parameter_use(f'./result/pkl/prototype_{prototype_num}/acc_once_epoch50_not_finetune_ae/train_model_epoch100_{prototype_num}.pkl').to(device)
optimizer = optim.Adam(net.parameters(), lr=0.002)

# training parameters
num_epochs = 750
test_display_step = 50
save_step = 50
pruning_step = 50
final_pruning_step = 50

# elastic deformation parameters
sigma = 4
alpha = 20

weight_matrix = net.classifier[0].weight
de_mask = torch.ones(weight_matrix.shape)
proto_matrix = net.prototype_feature_vectors
proto_mask = torch.ones(proto_matrix.shape)
prune_proto_list = list()
avg_test_acc = 0

# for param in net.parameters():
#     param.requires_grad = False
# net.prototype_feature_vectors.requires_grad = True
# for dense in net.classifier:
#     dense.weight.requires_grad = True
for epoch in range(num_epochs):
    # train
    net.train()
    train_loss, train_acc, train_class_error, train_ae_error, train_error_1, train_error_2 = 0, 0, 0, 0, 0, 0
    for i, (images, labels) in enumerate(train_loader):
        elastic_images = batch_elastic_transform(images.reshape(-1, in_height * in_width), sigma=sigma, alpha=alpha,
                                                 height=in_height, width=in_width) \
            .reshape(-1, in_channel_num, in_height, in_width)
        elastic_images, labels = torch.tensor(elastic_images, dtype=dtype).to(device), labels.to(device)
        optimizer.zero_grad()
        ae_output, prototype_distances, feature_vector_distances, outputs, softmax_output = net(elastic_images)
        class_error = criterion(outputs, labels)
        train_class_error += criterion(outputs, labels)
        ae_output = ae_output.reshape(-1, 1, in_width, in_height)  # simple_ae
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
    avg_train_class_error, avg_train_ae_error = train_class_error / len(train_loader.dataset), train_ae_error / len(
        train_loader.dataset)
    avg_train_error_1, avg_train_error_2 = train_error_1 / len(train_loader.dataset), train_error_2 / len(
        train_loader.dataset)

    # val
    net.eval()
    val_loss, val_acc, val_class_error, val_ae_error, val_error_1, val_error_2 = 0, 0, 0, 0, 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            ae_output, prototype_distances, feature_vector_distances, outputs, softmax_output = net(images)
            class_error = criterion(outputs, labels)
            val_class_error += criterion(outputs, labels)
            ae_output = ae_output.reshape(-1, 1, in_width, in_height)  # simple_ae
            ae_error = torch.mean(list_of_norms(ae_output - images))
            val_ae_error += torch.mean(list_of_norms(ae_output - images))
            error_1 = torch.mean(torch.min(feature_vector_distances))
            val_error_1 += torch.mean(torch.min(feature_vector_distances))
            error_2 = torch.mean(torch.min(prototype_distances))
            val_error_2 += torch.mean(torch.min(prototype_distances))
            loss = prototype_loss(val_class_error, val_ae_error, val_error_1, val_error_2, error_1_flag=True,
                                  error_2_flag=True)
            val_loss += loss.item()
            val_acc += (outputs.max(1)[1] == labels).sum().item()
        avg_val_loss, avg_val_acc = val_loss / len(val_loader.dataset), val_acc / len(val_loader.dataset)
        avg_val_class_error, avg_val_ae_error = val_class_error / len(val_loader.dataset), val_ae_error / len(
            val_loader.dataset)
        avg_val_error_1, avg_val_error_2 = val_error_1 / len(val_loader.dataset), val_error_2 / len(val_loader.dataset)

    print(f'epoch [{epoch + 1}/{num_epochs}], train_loss: {avg_train_loss:.4f}'
          f', train_acc: {avg_train_acc:.4f}, 'f'val_loss: {avg_val_loss:.4f}, val_acc: {avg_val_acc:.4f}')

    # test
    avg_test_acc, test_acc = 0, 0
    with torch.no_grad():
        # if epoch % test_display_step == 0 or epoch == num_epochs - 1:
        if ((epoch <= 200 and epoch % pruning_step == 0) or (epoch > 200 and epoch % final_pruning_step == 0)) or epoch == num_epochs - 1:
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                ae_output, prototype_distances, feature_vector_distances, outputs, softmax_output = net(images)
                test_acc += (outputs.max(1)[1] == labels).sum().item()
            avg_test_acc = test_acc / len(test_loader.dataset)
            print(f'test_acc: {avg_test_acc:.4f}')

    # save the learning history
    learning_history['epoch'].append(epoch + 1)
    learning_history['train_acc'].append(f'{avg_train_acc:.4f}')
    learning_history['train_total_loss'].append(f'{avg_train_loss:.4f}')
    learning_history['train_class_loss'].append(f'{avg_train_class_error:.4e}')
    learning_history['train_ae_loss'].append(f'{avg_train_ae_error:.4e}')
    learning_history['train_error_1_loss'].append(f'{avg_train_error_1:.4e}')
    learning_history['train_error_2_loss'].append(f'{avg_train_error_2:.4e}')
    learning_history['val_acc'].append(f'{avg_val_acc:.4f}')
    learning_history['val_total_loss'].append(f'{avg_val_loss:.4f}')
    learning_history['val_class_loss'].append(f'{avg_val_class_error:.4e}')
    learning_history['val_ae_loss'].append(f'{avg_val_ae_error:.4e}')
    learning_history['val_error_1_loss'].append(f'{avg_val_error_1:.4e}')
    learning_history['val_error_2_loss'].append(f'{avg_val_error_2:.4e}')
    learning_history['test_acc'].append(f'{avg_test_acc:.4f}')
    result_save(f'./result/csv/train_history_{prototype_num}.csv', learning_history)

    # save model, prototype and ae_out
    if epoch % save_step == 0 or epoch == num_epochs - 1 or (epoch > 200 and epoch % final_pruning_step == 0):
        with torch.no_grad():
            parameter_save(f'./result/pkl/train_model_epoch{epoch + 1}_{prototype_num}.pkl', net)

            f_width = int(math.sqrt(len(net.prototype_feature_vectors[1]) / class_num))
            f_height = int(math.sqrt(len(net.prototype_feature_vectors[1]) / class_num))
            prototype_imgs = net.decoder(
                # prototype_imgs = net.cifar_decoder(
                # prototype_imgs = net.simple_decoder(
                net.prototype_feature_vectors.reshape(prototype_num, class_num, f_width, f_height)).cpu().numpy()
            # net.prototype_feature_vectors).cpu().numpy()
            n_cols = 5
            n_rows = prototype_num // n_cols + 1 if prototype_num % n_cols != 0 else prototype_num // n_cols
            g, b = plt.subplots(n_rows, n_cols, figsize=(n_cols, n_rows), squeeze=False)
            for i in range(n_rows):
                for j in range(n_cols):
                    if i * n_cols + j < prototype_num:
                        b[i][j].imshow(
                            prototype_imgs[i * n_cols + j].reshape(in_height, in_width),
                            # prototype_imgs[i * n_cols + j].reshape(in_height, in_width, in_channel_num),
                            cmap='gray',
                            interpolation='none')
                        b[i][j].axis('off')
            plt.savefig(f'./result/png/prototype_epoch{epoch + 1}_{prototype_num}.png',
                        transparent=True, bbox_inches='tight', pad_inches=0)
            plt.close()

            examples_to_show = 10
            examples = [train_dataset[i][0] for i in range(examples_to_show)]
            examples = torch.cat(examples).reshape(len(examples), *examples[0].shape).to(device)
            # examples = torch.cat(examples).reshape(len(examples), in_width * in_height).to(device)  # simple_decoder
            encode_decode = net.decoder(net.encoder(examples))
            # encode_decode = net.cifar_decoder(net.cifar_encoder(examples))
            # encode_decode = net.simple_decoder(net.simple_encoder(examples))
            f, a = plt.subplots(2, examples_to_show, figsize=(examples_to_show, 2), squeeze=False)
            for i in range(examples_to_show):
                a[0][i].imshow(
                    examples[i].cpu().numpy().reshape(in_height, in_width),
                    # examples[i].cpu().numpy().reshape(in_height, in_width, in_channel_num),
                    cmap='gray',
                    interpolation='none')
                a[0][i].axis('off')
                a[1][i].imshow(
                    encode_decode[i].cpu().numpy().reshape(in_height, in_width),
                    # encode_decode[i].cpu().numpy().reshape(in_height, in_width, in_channel_num),
                    cmap='gray',
                    interpolation='none')
                a[1][i].axis('off')
            plt.savefig(f'./result/png/ae_decode_epoch{epoch + 1}_{prototype_num}.png',
                        transparent=True, bbox_inches='tight', pad_inches=0)
            plt.close()

    # prune the prototypes that are close to other prototypes
    with torch.no_grad():
        if epoch != 0 and epoch <= 250 and epoch % pruning_step == 0:
        # if epoch != 0 and ((epoch <= 200 and epoch % pruning_step == 0) or (epoch > 200 and epoch % final_pruning_step == 0)):
            proto_distance_list = list()
            for i in range(prototype_num):
                tmp_distance_sum = 0
                for j in range(prototype_num):
                    if j not in prune_proto_list:
                        tmp_distance_sum += euclidean(net.prototype_feature_vectors[i].cpu().detach().numpy(),
                                                      net.prototype_feature_vectors[j].cpu().detach().numpy())
                # Exclude prototypes that have already been pruned
                if i in prune_proto_list:
                    tmp_distance_sum = 10000000
                proto_distance_list.append(tmp_distance_sum)
            if epoch <= 200:
                tmp_prune_proto_list = np.argsort(np.array(proto_distance_list))[:20]
            # when prototype_num = 100
            elif epoch <= 250:
                tmp_prune_proto_list = np.argsort(np.array(proto_distance_list))[:10]
            # Add already pruned prototype
            prune_proto_list.extend(tmp_prune_proto_list)
            for idx in tmp_prune_proto_list:
                print(f'\n{idx + 1}th prototype pruning')
                de_mask[:, idx] = 0
                proto_mask[idx] = 0
            weight_matrix *= torch.tensor(de_mask, device=device, dtype=dtype)
            proto_matrix *= torch.tensor(proto_mask, device=device, dtype=dtype)
            # only near prototype pruned
            # elif epoch > 200:
            #     tmp_prune_proto_list = np.argsort(np.array(proto_distance_list))[0]
            #     # Add already pruned prototype
            #     prune_proto_list.append(tmp_prune_proto_list)
            #     print(f'\n{tmp_prune_proto_list + 1}th prototype pruning')
            #     de_mask[:, tmp_prune_proto_list] = 0
            #     proto_mask[tmp_prune_proto_list] = 0
            #     weight_matrix *= torch.tensor(de_mask, device=device, dtype=dtype)
            #     proto_matrix *= torch.tensor(proto_mask, device=device, dtype=dtype)

        # pruning the prototypes that does not lose accuracy even if pruned
        if epoch > 250 and (epoch % final_pruning_step == 0 or epoch == num_epochs - 1):
            tmp_prune_proto_list = list()
            tmp_accuracy_dict = {}
            tmp_weight_matrix = weight_matrix.clone()
            for idx in range(prototype_num):
                if idx not in prune_proto_list:
                    tmp_de_mask = torch.ones(weight_matrix.shape)
                    tmp_de_mask[:, idx] = 0
                    weight_matrix *= torch.tensor(tmp_de_mask, device=device, dtype=dtype)
                    tmp_test_acc = 0
                    for images, labels in test_loader:
                        images, labels = images.to(device), labels.to(device)
                        ae_output, prototype_distances, feature_vector_distances, outputs, softmax_output = net(images)
                        tmp_test_acc += (outputs.max(1)[1] == labels).sum().item()
                    tmp_avg_test_acc = tmp_test_acc / len(test_loader.dataset)
                    # Pruning prototype does not reduce accuracy below the threshold------------------------------------
                    # if tmp_avg_test_acc > avg_test_acc - 0.5:
                    #     tmp_prune_proto_list.append(idx)
                    #     prune_proto_list.append(idx)
                    #     print(f'\n{idx + 1}th prototype pruning')
                    # weight_matrix = tmp_weight_matrix.clone()
                    # net.classifier[0].weight.data = weight_matrix
                    # Pruning prototype that reduce the accuracy least--------------------------------------------------
                    tmp_accuracy_dict[idx] = tmp_avg_test_acc
                    weight_matrix = tmp_weight_matrix.clone()
                    net.classifier[0].weight.data = weight_matrix
            tmp_prune_proto_list.append(max(tmp_accuracy_dict, key=tmp_accuracy_dict.get))
            prune_proto_list.append(max(tmp_accuracy_dict, key=tmp_accuracy_dict.get))
            print(f'\n{max(tmp_accuracy_dict, key=tmp_accuracy_dict.get) + 1}th prototype pruning')
            for i in tmp_prune_proto_list:
                de_mask[:, i] = 0
                proto_mask[i] = 0
            weight_matrix *= torch.tensor(de_mask, device=device, dtype=dtype)
            proto_matrix *= torch.tensor(proto_mask, device=device, dtype=dtype)

testacc_vis(f'./result/png/testacc.png', num_epochs % test_display_step, learning_history)
