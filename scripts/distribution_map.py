import os
import sys
pardir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(pardir)
from util_func import parameter_use, parameter_save, outlier_2s
from dataset import *
import model
from model import class_num, prototype_num
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import manifold
from adjustText import adjust_text
import umap
import umap.plot
from scipy.sparse.csgraph import connected_components
from matplotlib.gridspec import GridSpec
import math

prototype = "15"
offset = 0
for k in range(2, 3):
    # train_net = parameter_use(f'./result/pkl/prototype_{prototype}/train_model_epoch500_{prototype}.pkl')
    # train_net = parameter_use(f'./result/pkl/3NN_prototype_{prototype}/train_model_epoch500_{prototype}.pkl')
    # train_net = parameter_use(f'./result/pkl/prototype_{prototype}/prune_dense/not_abs/prune_proto_finetune_from_small/'
    #                           f'prune_train_model_epoch{k}_{prototype}.pkl')
    # train_net = parameter_use(f'./result/pkl/prototype_{prototype}/prune_dense/not_abs/prune_all_finetune_from_small_at_least_1weight/'
    #                           f'prune_train_model_epoch{k}_{prototype}.pkl')
    # train_net = parameter_use(f'./result/pkl/prototype_{prototype}/prune_dense/prune_finetune_once/prune_negative/'
    #                           f'prune_train_model_epoch2_{prototype}.pkl')
    # if offset <= 700:
    #     train_net = parameter_use(f'./result/pkl/prototype_{prototype}/acc_once_epoch50/train_model_epoch{offset+1}_{prototype}.pkl')
    # elif offset == 750:
    #     train_net = parameter_use(f'./result/pkl/prototype_{prototype}/acc_once_epoch50/train_model_epoch{offset}_{prototype}.pkl')
    # if offset <= 750:
    #     offset += 50
    # elif offset > 150:
    #     offset += 10
    # train_net = parameter_use(f'./result/pkl/fashionmnist_prototype{prototype}/prune_dense/abs/prune_conv_finetune_from_abs_large/'
    #                           f'prune_train_model_epoch{k}_{prototype}.pkl')
    # train_net = parameter_use(f'./result/pkl/prototype_{prototype}/prune_proto/prune_all_finetune_from_near_epoch30/'
    #                           f'prune_train_model_epoch{k}_{prototype}.pkl')
    train_net = parameter_use(f'./result/pkl/fashionmnist_prototype{prototype}/rotate/train_model_epoch500_{prototype}.pkl')

    examples_to_show = 10000
    examples = [test_dataset[i][0] for i in range(examples_to_show)]
    examples = torch.cat(examples).reshape(len(examples), *examples[0].shape).to(device)
    examples_labels = [test_dataset[i][1] for i in range(examples_to_show)]
    examples_class_labels = examples_labels.copy()
    # Fashion MNIST
    # examples_class_labels_sub = examples_labels.copy()
    # examples_class_labels = list()
    # for item in examples_class_labels_sub:
    #     item_mod = ''
    #     if item == 0:
    #         item_mod = "T-shirt/top"
    #     elif item == 1:
    #         item_mod = "Trouser"
    #     elif item == 2:
    #         item_mod = "Pullover"
    #     elif item == 3:
    #         item_mod = "Dress"
    #     elif item == 4:
    #         item_mod = "Coat"
    #     elif item == 5:
    #         item_mod = "Sandal"
    #     elif item == 6:
    #         item_mod = "Shirt"
    #     elif item == 7:
    #         item_mod = "Sneaker"
    #     elif item == 8:
    #         item_mod = "Bag"
    #     elif item == 9:
    #         item_mod = "Ankle boot"
    #     examples_class_labels.append(item_mod)
    color_set = ["b", "g", "r", "c", "m", "y", "#6a5acd", '#f781bf', '#a65628', '#ff7f00']
    examples_labels = np.where(examples_labels == "0", "b", examples_labels)
    for i in range(model.class_num):
        examples_labels = np.where(examples_labels == str(i), color_set[i], examples_labels)
    feature_vec = train_net.encoder(examples).reshape(-1, model.class_num * 2 * 2).cpu().detach().numpy()
    # feature_vec = train_net.encoder(examples).reshape(-1, model.class_num).cpu().detach().numpy()
    prototype_feature_vec = train_net.prototype_feature_vectors.reshape(-1, model.class_num * 2 * 2).cpu().detach().numpy()
    # prototype_feature_vec = train_net.prototype_feature_vectors.reshape(-1, model.class_num).cpu().detach().numpy()
    prototype_feature_vec_2 = prototype_feature_vec.copy()
    pruned_prototype = np.sum(prototype_feature_vec, axis=1)
    prune_count = 0
    pruned_list = list()
    for i, value in enumerate(pruned_prototype):
        if abs(value) < 0.00001:
            prune_count += 1
            pruned_list.append(i)
    prototype_feature_vec_2 = np.delete(prototype_feature_vec.copy(), pruned_list, 0)
    con_vec = np.concatenate([feature_vec, prototype_feature_vec_2])

    # pca---------------------------------------------------
    # pca = PCA(n_components=2)
    # pca.fit(con_vec)
    # data_pca = pca.transform(con_vec)
    # t-sne-------------------------------------------------
    # pca = manifold.TSNE(n_components=2, init='pca', random_state=0)
    # data_pca = pca.fit_transform(feature_vec)
    # umap--------------------------------------------------
    pca = umap.UMAP()
    data_pca = pca.fit_transform(con_vec)
    # parameter_save(f'./result/pkl/umap_prototype{prototype_num}.pkl', data_pca)
    # data_pca = parameter_use(f'./result/pkl/prototype_{prototype}/umap/umap_prototype{prototype_num}.pkl')
    # labels = np.array([test_dataset[i][1] for i in range(examples_to_show)])

    fig = plt.figure(figsize=(15, 12), facecolor='w', tight_layout=True)
    plt.rcParams["font.size"] = 15
    # plt.tick_params(labelbottom=False, labelleft=False)
    prototype_name = [chr(ord('A') + i) for i in range(int(prototype))]
    j = 0
    handles, label_keep, texts = [], [], []
    for i in range(examples_to_show + int(prototype) - prune_count):
        # plot data
        if i < examples_to_show:
            line, = plt.plot(data_pca[i][0], data_pca[i][1], ms=5.0, zorder=2, marker="x",
                             color=examples_labels[i], label=examples_class_labels[i])
            plt.tick_params(labelsize=38)
            if not (examples_labels[i] in label_keep):
                label_keep.append(examples_labels[i])
                handles.append(line)
        # plot prototype
        else:
            plt.plot(data_pca[i][0], data_pca[i][1], ms=50.0, zorder=2, marker=".", color='gray')
            # show prototype order
            plt_text = plt.annotate(prototype_name[j], (data_pca[i][0], data_pca[i][1]), size=60)
            texts.append(plt_text)
            j += 1
    # Specify the shape and color of the line pointing to the point plotted from the label with arrowprops
    adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=7))
    # legend_order = [color_set.index(val) for val in label_keep]
    # handles_order = [legend_order.index(val) for val in range(model.class_num)]
    # sorted_handles = [handles[idx] for idx in handles_order]
    # fig.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=1, fontsize=25, handles=sorted_handles, edgecolor='black')

    # plt.savefig(f'./result/png/prototype_{prototype}/distribution_map.png')
    # plt.savefig(f'./result/png/3NN_prototype_{prototype}/distribution_map.png')
    # plt.savefig(f'./result/png/prototype_{prototype}/prune_dense/prune_finetune/not_abs/prune{k}_distribution_map_nolegend.png')
    # plt.savefig(f'./result/png/prototype_{prototype}/prune_finetune_once/prune{k}_distribution_map.png')
    # if offset <= 750:
    #     plt.savefig(f'./result/png/prototype_{prototype}/epoch{offset - 50 + 1}_prune_distribution_map.png')
    # elif 200 < offset <= 300:
    #     plt.savefig(f'./result/png/prototype_{prototype}/epoch{offset - 10 + 1}_prune_distribution_map.png')
    # elif offset == 800:
    #     plt.savefig(f'./result/png/prototype_{prototype}/epoch{offset - 50}_prune_distribution_map.png')
    # plt.savefig(f'./result/png/fashionmnist_prototype{prototype}/prune_finetune/abs/prune{k}_distribution_map.png')
    # plt.savefig(f'./result/png/prototype_{prototype}/prune_proto/prune_finetune/prune{k}_distribution_map.png')
    plt.savefig(f'./result/png/fashionmnist_prototype{prototype}/rotate/distribution_map.png')
    # plt.show()
