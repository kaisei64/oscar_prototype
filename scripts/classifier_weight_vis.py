import os
import sys
pardir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(pardir)
from util_func import parameter_use, weight_distribution_vis
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

prototype = "15"
for k in range(2, 11):
    # train_net = parameter_use(f'./result/pkl/prototype_{prototype}/prune_train_model_epoch500_{prototype}.pkl')
    # train_net = parameter_use(f'./result/pkl/3NN_prototype_{prototype}/train_model_epoch500_{prototype}.pkl')
    # train_net = parameter_use(f'./result/pkl/prototype_{prototype}/prune_dense/not_abs/prune_all_finetune_from_small_at_least_1weight/'
    #                           f'prune_train_model_epoch2_{prototype}.pkl')
    # train_net = parameter_use(f'./result/pkl/prototype_{prototype}/prune_dense/prune_finetune_once/prune_negative/'
    #                           f'prune_train_model_epoch2_15.pkl')
    # train_net = parameter_use(f'./result/pkl/prototype_{prototype}/acc_10_denseis0/train_model_epoch500_{prototype}.pkl')
    train_net = parameter_use(f'./result/pkl/fashionmnist_prototype{prototype}/prune_dense/abs/prune_all_finetune_from_abs_small/'
                              f'prune_train_model_epoch{k}_{prototype}.pkl')

    classifier_weight = train_net.classifier[0].weight.cpu().detach().numpy().T

    class_name = [i for i in range(10)]
    prototype_name = [chr(ord('A') + i) for i in range(int(prototype))]
    df = pd.DataFrame(classifier_weight, columns=class_name)
    # print(df)
    fig, ax = plt.subplots(figsize=((len(df.columns))*1.2, (len(df))*0.4))
    ax.axis('off')
    # Color the weights that contribute a lot
    colors = [["silver" if classifier_weight[j][i] - min(classifier_weight[j]) < abs(min(classifier_weight[j])) * 0.4
               else "w" for i in range(len(classifier_weight[0]))]
              for j in range(len(classifier_weight[:, 0]))]
    vals = np.around(df.values, 2)
    normal = plt.Normalize(vals.min()-1, vals.max()+1)
    tbl = ax.table(cellText=df.values.round(3),
                   bbox=[0.05, 0, 0.9, 1],
                   colLabels=df.columns,
                   rowLabels=prototype_name,
                   cellColours=colors)
                   # cellColours=plt.cm.gray(normal(vals)))
    tbl.set_fontsize(30)
    # plt.savefig(f'./result/png/prototype_{prototype}/classifier_weight.png')
    # plt.savefig(f'./result/png/3NN_prototype_{prototype}/classifier_weight.png')
    # plt.savefig(f'./result/png/prototype_{prototype}/prune_finetune/not_abs/prune2_classifier_weight.png')
    # plt.savefig(f'./result/png/prototype_{prototype}/acc_10_denseis0/proto_prune_classifier_weight.png')
    plt.savefig(f'./result/png/fashionmnist_prototype{prototype}/prune_finetune/abs/prune{k}_classifier_weight.png')
    # plt.show()

# classifier weight distribution
# for i in range(1, 11):
#     train_net = parameter_use(f'./result/pkl/prototype_{prototype}/prune_dense/not_abs/prune_proto_finetune_from_small/'
#                               f'prune_train_model_epoch{i}_{prototype}.pkl')
#     classifier_weight = train_net.classifier[0].weight.cpu().detach().numpy().flatten()
#     weight_distribution_vis(f'./result/png/prototype_{prototype}/prune_finetune/not_abs/prune{i}_classifier_weight_dis.png', classifier_weight)

# conv weight distribution
# for i in range(1, 11):
#     train_net = parameter_use(f'./result/pkl/prototype_{prototype}/prune_dense/abs/prune_proto_finetune_from_abs_small/'
#                               f'prune_train_model_epoch{i}_{prototype}.pkl')
#     conv_list = [module for module in train_net.modules() if isinstance(module, nn.Conv2d)]
#     for j, conv in enumerate(conv_list):
#         conv_weight = conv.weight.cpu().detach().numpy().flatten()
#         weight_distribution_vis(f'./result/png/prototype_{prototype}/prune_finetune/abs/prune{i}_'
#                                 f'conv{j}_weight_dis.png', conv_weight)
