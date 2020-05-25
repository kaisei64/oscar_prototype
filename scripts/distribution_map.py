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

examples_to_show = 10000
examples = [train_dataset[i][0] for i in range(examples_to_show)]
examples = torch.cat(examples).reshape(len(examples), *examples[0].shape).to(device)
examples_labels = [train_dataset[i][1] for i in range(examples_to_show)]
color_set = ["b", "g", "r", "c", "m", "y", "k", '#f781bf', '#a65628', '#ff7f00']
for i in range(model.class_num):
    examples_labels = np.where(examples_labels == str(i), color_set[i], examples_labels)
feature_vec = train_net.encoder(examples).reshape(-1, model.class_num*2*2).cpu().detach().numpy()
pca = PCA(n_components=2)
pca.fit(feature_vec)
data_pca = pca.transform(feature_vec)
fig = plt.figure(figsize=(15, 12), facecolor='w')
plt.rcParams["font.size"] = 15
for i in range(examples_to_show):
    # plot data
    plt.plot(data_pca[i][0], data_pca[i][1], ms=5.0, zorder=2, marker="x", color=examples_labels[i])
    # show label
    # plt.annotate(words[i][0], (data_pca[i][0], data_pca[i][1]), size=12)
plt.show()
