import os
import sys
pardir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(pardir)
from util_func import parameter_use, batch_elastic_transform, list_of_norms, result_save, parameter_save, outlier_2s
from loss import *
from dataset import *
from model import class_num, in_channel_num, prototype_num
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib.gridspec import GridSpec

learning_history = {'epoch': [], 'train_acc': [], 'train_total_loss': [], 'train_class_loss': [], 'train_ae_loss': [],
                    'train_error_1_loss': [], 'train_error_2_loss': [], 'val_acc': [], 'val_total_loss': [],
                    'val_class_loss': [], 'val_ae_loss': [], 'val_error_1_loss': [], 'val_error_2_loss': [],
                    'test_acc': []}

prototype = "15"
train_net = parameter_use(f'./result/pkl/prototype_{prototype}/train_model_epoch500_{prototype}.pkl')
optimizer = optim.Adam(train_net.parameters(), lr=0.002)
# training parameters
num_epochs = 300
save_step = 50
test_display_step = 50
# elastic deformation parameters
sigma = 4
alpha = 20

# make prototype----------------------------------------------------------------------------------------------------
train_net = parameter_use(f'./result/pkl/prototype_{prototype}/train_model_epoch500_{prototype}.pkl')
examples_to_show = 10000
color_set = ["b", "g", "r", "c", "m", "y", "#6a5acd", '#f781bf', '#a65628', '#ff7f00']
prototype_name = [chr(ord('A') + i) for i in range(int(prototype))]

# pca = umap.UMAP()
# data_pca = pca.fit(feature_vec)
# parameter_save(f'./result/pkl/umap_prototype{prototype_num}.pkl', data_pca)
data_pca = parameter_use(f'./result/pkl/prototype_{prototype}/umap/umap_prototype{prototype_num}.pkl')
labels = np.array([test_dataset[i][1] for i in range(examples_to_show)])

embed = data_pca.embedding_
embed_labels = list()
for i, label in enumerate(labels):
    tmp = np.append(embed[i], label)
    if i == 0:
        embed_labels = tmp
    else:
        embed_labels = np.vstack([embed_labels, tmp])
embed_list = list()
for i in range(class_num):
    embed_list.append([x[:2] for x in embed_labels if x[2] == i])

top_left = np.array([np.amin(embed, axis=0)[0], np.amax(embed, axis=0)[1]])
btm_left = np.array([np.amin(embed, axis=0)[0], np.amin(embed, axis=0)[1]])
top_right = np.array([np.amax(embed, axis=0)[0], np.amax(embed, axis=0)[1]])
btm_right = np.array([np.amax(embed, axis=0)[0], np.amin(embed, axis=0)[1]])

test_pts = np.array([
    (top_left * (1 - x) + top_right * x) * (1 - y) +
    (btm_left * (1 - x) + btm_right * x) * y
    for y in np.linspace(0, 1, 10)
    for x in np.linspace(0, 1, 10)
])

# excluding out of ranges
embed_list_copy = embed_list.copy()
for i, tmp_emb in enumerate(embed_list):
    dim1_min_value, dim1_max_value = outlier_2s(np.array(tmp_emb)[:, 0])
    dim2_min_value, dim2_max_value = outlier_2s(np.array(tmp_emb)[:, 1])
    delete_list = list()
    for j, value in enumerate(tmp_emb):
        if value[0] < dim1_min_value or value[0] > dim1_max_value or value[1] < dim2_min_value or value[1] > dim2_max_value:
            delete_list.append(j)
    embed_list[i] = np.delete(embed_list[i], delete_list, 0)

border_point_015 = np.array([np.array([np.amax(embed_list[0], axis=0)[0], np.amax(embed_list[0], axis=0)[1]])
                             + np.array([np.amin(embed_list[1], axis=0)[0], np.amax(embed_list[1], axis=0)[1]])
                             + np.array([np.amin(embed_list[5], axis=0)[0], np.amin(embed_list[5], axis=0)[1]])]) / 3
border_point_016 = np.array([np.array([np.amax(embed_list[0], axis=0)[0], np.amin(embed_list[0], axis=0)[1]])
                             + np.array([np.amin(embed_list[1], axis=0)[0], np.amin(embed_list[1], axis=0)[1]])
                             + np.array([np.amax(embed_list[6], axis=0)[0], np.amax(embed_list[6], axis=0)[1]])]) / 3
border_point_135 = np.array([np.array([np.amax(embed_list[1], axis=0)[0], np.amax(embed_list[1], axis=0)[1]])
                             + np.array([np.amin(embed_list[3], axis=0)[0], np.amin(embed_list[3], axis=0)[1]])
                             + np.array([np.amax(embed_list[5], axis=0)[0], np.amin(embed_list[5], axis=0)[1]])]) / 3
border_point_127 = np.array([np.array([np.amax(embed_list[1], axis=0)[0], np.mean(embed_list[1], axis=0)[1]])
                             + np.array([np.mean(embed_list[2], axis=0)[0], np.amax(embed_list[2], axis=0)[1]])
                             + np.array([np.amin(embed_list[7], axis=0)[0], np.amin(embed_list[7], axis=0)[1]])]) / 3
border_point_137 = np.array([np.array([np.amax(embed_list[1], axis=0)[0], np.amax(embed_list[1], axis=0)[1]])
                             + np.array([np.mean(embed_list[3], axis=0)[0], np.amin(embed_list[3], axis=0)[1]])
                             + np.array([np.amin(embed_list[7], axis=0)[0], np.amin(embed_list[7], axis=0)[1]])]) / 3
border_point_168 = np.array([np.array([np.amin(embed_list[1], axis=0)[0], np.amin(embed_list[1], axis=0)[1]])
                             + np.array([np.amax(embed_list[6], axis=0)[0], np.amax(embed_list[6], axis=0)[1]])
                             + np.array([np.amin(embed_list[8], axis=0)[0], np.amax(embed_list[8], axis=0)[1]])]) / 3
border_point_128 = np.array([np.array([np.mean(embed_list[1], axis=0)[0], np.amin(embed_list[1], axis=0)[1]])
                             + np.array([np.amin(embed_list[2], axis=0)[0], np.amin(embed_list[2], axis=0)[1]])
                             + np.array([np.amax(embed_list[8], axis=0)[0], np.amax(embed_list[8], axis=0)[1]])]) / 3
border_point_248 = np.array([np.array([np.amin(embed_list[2], axis=0)[0], np.amin(embed_list[2], axis=0)[1]])
                             + np.array([np.mean(embed_list[4], axis=0)[0], np.amax(embed_list[4], axis=0)[1]])
                             + np.array([np.amax(embed_list[8], axis=0)[0], np.amax(embed_list[8], axis=0)[1]])]) / 3
border_point_249 = np.array([np.array([np.amin(embed_list[2], axis=0)[0], np.amin(embed_list[2], axis=0)[1]])
                             + np.array([np.amax(embed_list[4], axis=0)[0], np.amax(embed_list[4], axis=0)[1]])
                             + np.array([np.amin(embed_list[9], axis=0)[0], np.mean(embed_list[9], axis=0)[1]])]) / 3
border_point_279 = np.array([np.array([np.amax(embed_list[2], axis=0)[0], np.mean(embed_list[2], axis=0)[1]])
                             + np.array([np.mean(embed_list[7], axis=0)[0], np.amin(embed_list[7], axis=0)[1]])
                             + np.array([np.mean(embed_list[9], axis=0)[0], np.amax(embed_list[9], axis=0)[1]])]) / 3
border_point_list = np.array([border_point_015, border_point_016, border_point_135, border_point_127,
                              border_point_137, border_point_168, border_point_128, border_point_248,
                              border_point_249, border_point_279]).squeeze()

f_width = int(math.sqrt(len(train_net.prototype_feature_vectors[1]) / class_num))
f_height = int(math.sqrt(len(train_net.prototype_feature_vectors[1]) / class_num))
# test_point
inv_transformed_points = train_net.decoder(torch.tensor(data_pca.inverse_transform(test_pts).reshape(100
                                                                                                     , class_num,
                                                                                                     f_width,
                                                                                                     f_height),
                                                        dtype=dtype, device=device)).cpu().detach().numpy()
# border_point
inv_border_points = torch.tensor(data_pca.inverse_transform(border_point_list), dtype=dtype, device=device)
inv_transformed_border_points = train_net.decoder(
    inv_border_points.reshape(len(border_point_list), class_num, f_width, f_height)).cpu().detach().numpy()

# Set up the grid
fig = plt.figure(figsize=(12, 6), facecolor='w', tight_layout=True)
# gs = GridSpec(10, 20, fig)
gs = GridSpec(2, 10, fig)
# scatter_ax = fig.add_subplot(gs[:, :10])
scatter_ax = fig.add_subplot(gs[:, :5])
digit_axes = np.zeros((10, 10), dtype=object)
border_axes = np.zeros((2, 5), dtype=object)
# for i in range(10):
#     for j in range(10):
#         digit_axes[i, j] = fig.add_subplot(gs[i, 10 + j])
for i in range(2):
    for j in range(5):
        border_axes[i, j] = fig.add_subplot(gs[i, 5 + j])

# Use umap.plot to plot to the major axis
# scatter_ax.scatter(embed[:, 0], embed[:, 1], c=labels.astype(np.int32), cmap='Spectral', s=0.1)
i = 0
for color, tmp_embed in zip(color_set, embed_list_copy):
    scatter_ax.scatter(np.array(tmp_embed)[:, 0], np.array(tmp_embed)[:, 1], c=color, s=10, label=str(i))
    i += 1
scatter_ax.set(xticks=[], yticks=[])
# fig.legend(bbox_to_anchor=(1, 1), loc='upper left', fontsize=25, edgecolor='black')

# Plot the locations of the text points
# scatter_ax.scatter(test_pts[:, 0], test_pts[:, 1], marker='x', c='k', s=15)
scatter_ax.scatter(border_point_list[:, 0], border_point_list[:, 1], marker='.', c='gray', s=500)
texts = list()
for i in range(len(border_point_list)):
    text = scatter_ax.annotate(prototype_name[i], (border_point_list[i][0], border_point_list[i][1]), size=20)
    # texts.append(text)
# adjust_text(texts)

# Plot each of the generated digit images
# for i in range(10):
#     for j in range(10):
#         digit_axes[i, j].imshow(inv_transformed_points[i * 10 + j].reshape(28, 28))
#         digit_axes[i, j].set(xticks=[], yticks=[])
for i in range(2):
    for j in range(5):
        border_axes[i, j].imshow(inv_transformed_border_points[i * 5 + j].reshape(28, 28))
parameter_save(f'./result/pkl/prototype_{prototype}/border/border_point.pkl', inv_border_points)
plt.savefig(f'./result/png/prototype_{prototype}/distribution_map.png')

# use prototype-----------------------------------------------------------------------------------------------------
border_prototype_feature_vec = parameter_use(f'./result/pkl/prototype_{prototype}/border/border_point.pkl')
proto_matrix = train_net.prototype_feature_vectors
proto_mask = torch.ones(proto_matrix.shape)
with torch.no_grad():
    for i, value in enumerate(border_prototype_feature_vec):
        proto_matrix[i] = value
    proto_mask[len(border_prototype_feature_vec):, :] = 0
    proto_matrix *= torch.tensor(proto_mask, device=device, dtype=dtype)

for param in train_net.parameters():
    param.requires_grad = False
# train_net.prototype_feature_vectors.requires_grad = True
for dense in train_net.classifier:
    dense.weight.requires_grad = True
for epoch in range(num_epochs):
    # train
    train_net.train()
    train_loss, train_acc, train_class_error, train_ae_error, train_error_1, train_error_2 = 0, 0, 0, 0, 0, 0
    for i, (images, labels) in enumerate(train_loader):
        elastic_images = batch_elastic_transform(images.reshape(-1, in_height * in_width), sigma=sigma, alpha=alpha,
                                                 height=in_height, width=in_width) \
            .reshape(-1, in_channel_num, in_height, in_width)
        elastic_images, labels = torch.tensor(elastic_images, dtype=dtype).to(device), labels.to(device)
        optimizer.zero_grad()
        ae_output, prototype_distances, feature_vector_distances, outputs, softmax_output = train_net(elastic_images)
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
            proto_matrix *= torch.tensor(proto_mask, device=device, dtype=dtype)
    avg_train_loss, avg_train_acc = train_loss / len(train_loader.dataset), train_acc / len(train_loader.dataset)
    avg_train_class_error, avg_train_ae_error = train_class_error / len(train_loader.dataset), train_ae_error / len(train_loader.dataset)
    avg_train_error_1, avg_train_error_2 = train_error_1 / len(train_loader.dataset), train_error_2 / len(train_loader.dataset)

    # val
    train_net.eval()
    val_loss, val_acc, val_class_error, val_ae_error, val_error_1, val_error_2 = 0, 0, 0, 0, 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            ae_output, prototype_distances, feature_vector_distances, outputs, softmax_output = train_net(images)
            class_error = criterion(outputs, labels)
            val_class_error += criterion(outputs, labels)
            ae_output = ae_output.reshape(-1, 1, in_width, in_height)  # simple_ae
            ae_error = torch.mean(list_of_norms(ae_output - images))
            val_ae_error += torch.mean(list_of_norms(ae_output - images))
            error_1 = torch.mean(torch.min(feature_vector_distances))
            val_error_1 += torch.mean(torch.min(feature_vector_distances))
            error_2 = torch.mean(torch.min(prototype_distances))
            val_error_2 += torch.mean(torch.min(prototype_distances))
            loss = prototype_loss(val_class_error, val_ae_error, val_error_1, val_error_2, error_1_flag=True, error_2_flag=True)
            val_loss += loss.item()
            val_acc += (outputs.max(1)[1] == labels).sum().item()
        avg_val_loss, avg_val_acc = val_loss / len(val_loader.dataset), val_acc / len(val_loader.dataset)
        avg_val_class_error, avg_val_ae_error = val_class_error / len(val_loader.dataset), val_ae_error / len(val_loader.dataset)
        avg_val_error_1, avg_val_error_2 = val_error_1 / len(val_loader.dataset), val_error_2 / len(val_loader.dataset)
    print(f'epoch [{epoch + 1}/{num_epochs}], train_loss: {avg_train_loss:.4f}'
          f', train_acc: {avg_train_acc:.4f}, 'f'val_loss: {avg_val_loss:.4f}, val_acc: {avg_val_acc:.4f}')

    # test
    avg_test_acc, test_acc = 0, 0
    with torch.no_grad():
        if epoch % test_display_step == 0 or epoch == num_epochs - 1:
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                ae_output, prototype_distances, feature_vector_distances, outputs, softmax_output = train_net(images)
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
    result_save(f'./result/csv/train_history_{prototype_num}_border.csv', learning_history)

    # save model, prototype and ae_out
    if epoch % save_step == 0 or epoch == num_epochs - 1:
        with torch.no_grad():
            parameter_save(f'./result/pkl/train_model_epoch{epoch + 1}_{prototype_num}_border.pkl', train_net)

            f_width = int(math.sqrt(len(train_net.prototype_feature_vectors[1]) / class_num))
            f_height = int(math.sqrt(len(train_net.prototype_feature_vectors[1]) / class_num))
            prototype_imgs = train_net.decoder(
                # prototype_imgs = train_net.cifar_decoder(
                # prototype_imgs = train_net.simple_decoder(
                train_net.prototype_feature_vectors.reshape(prototype_num, class_num, f_width, f_height)).cpu().numpy()
            # train_net.prototype_feature_vectors).cpu().numpy()
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
            plt.savefig(f'./result/png/prototype_epoch{epoch + 1}_{prototype_num}_border.png',
                        transparent=True, bbox_inches='tight', pad_inches=0)
            plt.close()

            examples_to_show = 10
            examples = [train_dataset[i][0] for i in range(examples_to_show)]
            examples = torch.cat(examples).reshape(len(examples), *examples[0].shape).to(device)
            # examples = torch.cat(examples).reshape(len(examples), in_width * in_height).to(device)  # simple_decoder
            encode_decode = train_net.decoder(train_net.encoder(examples))
            # encode_decode = train_net.cifar_decoder(train_net.cifar_encoder(examples))
            # encode_decode = train_net.simple_decoder(train_net.simple_encoder(examples))
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
            plt.savefig(f'./result/png/ae_decode_epoch{epoch + 1}_{prototype_num}_border.png',
                        transparent=True, bbox_inches='tight', pad_inches=0)
            plt.close()
