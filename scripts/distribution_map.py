import os
import sys
pardir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(pardir)
from util_func import parameter_use
from dataset import *
import model
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

prototype = "15"
train_net = parameter_use(f'./result/pkl/prototype_{prototype}/train_model_1500.pkl')
# train_net = parameter_use(f'./result/pkl/prototype_{prototype}/train_model_epoch1500_{prototype}.pkl')
# train_net = parameter_use(f'./result/pkl/prototype_{prototype}_sub/train_model_epoch1500_{prototype}.pkl')

examples_to_show = 10000
examples = [train_dataset[i][0] for i in range(examples_to_show)]
examples = torch.cat(examples).reshape(len(examples), *examples[0].shape).to(device)
examples_labels = [train_dataset[i][1] for i in range(examples_to_show)]
color_set = ["b", "g", "r", "c", "m", "y", "k", '#f781bf', '#a65628', '#ff7f00']
examples_labels = np.where(examples_labels == "0", "b", examples_labels)
for i in range(model.class_num):
    examples_labels = np.where(examples_labels == str(i), color_set[i], examples_labels)
feature_vec = train_net.encoder(examples).reshape(-1, model.class_num*2*2).cpu().detach().numpy()
prototype_feature_vec = train_net.prototype_feature_vectors.reshape(-1, model.class_num*2*2)\
                        .cpu().detach().numpy()
con_vec = np.concatenate([feature_vec, prototype_feature_vec])

pca = PCA(n_components=2)
pca.fit(con_vec)
data_pca = pca.transform(con_vec)
fig = plt.figure(figsize=(15, 12), facecolor='w')
plt.rcParams["font.size"] = 15
num = [i for i in range(int(prototype))]
j = 0
for i in range(examples_to_show + int(prototype)):
    # plot data
    if i < examples_to_show:
        plt.plot(data_pca[i][0], data_pca[i][1], ms=5.0, zorder=2, marker="x", color=examples_labels[i])
    # plot prototype
    else:
        plt.plot(data_pca[i][0], data_pca[i][1], ms=50.0, zorder=2, marker=".", color='gray')
        # show prototype order
        plt.annotate(num[j], (data_pca[i][0], data_pca[i][1]), size=50)
        j += 1
plt.savefig(f'./result/png/prototype_{prototype}/distribution_map.png')
