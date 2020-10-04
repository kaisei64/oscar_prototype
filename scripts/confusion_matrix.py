import os
import sys
pardir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(pardir)
from util_func import parameter_use, make_confusion_matrix, pruningtestacc_vis
from dataset import *
import torch

prototype = "15"
offset = 0
# training parameters
num_epochs = 5
save_step = 10
# elastic deformation parameters
sigma = 4
alpha = 20

for k in range(2, 11):
    # train_net = parameter_use(f'./result/pkl/prototype_{prototype}/prune_dense/abs/prune_conv_finetune_from_abs_large/'
    #                           f'prune_train_model_epoch{k}_{prototype}.pkl')
    train_net = parameter_use(f'./result/pkl/fashionmnist_prototype{prototype}/prune_dense/abs/prune_proto_finetune_from_abs_large/'
                              f'prune_train_model_epoch{k}_{prototype}.pkl')
    # train_net = parameter_use(f'./result/pkl/prototype_{prototype}/prune_proto/prune_classifier_finetune_from_abs_large_epoch30/'
    #                           f'prune_train_model_epoch{k}_15.pkl')
    # if offset <= 700:
    #     train_net = parameter_use(f'./result/pkl/prototype_{prototype}/acc_once_epoch50/train_model_epoch{offset + 1}_{prototype}.pkl')
    # elif offset == 750:
    #     train_net = parameter_use(f'./result/pkl/prototype_{prototype}/acc_once_epoch50/train_model_epoch{offset}_{prototype}.pkl')
    # if offset <= 800:
    #     offset += 50
    # elif offset > 150:
    #     offset += 10

    output_hold = torch.empty(0)
    label_hold = torch.empty(0)
    with torch.no_grad():
        avg_test_acc, test_acc = 0, 0
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            ae_output, prototype_distances, feature_vector_distances, outputs, softmax_output = train_net(images)
            test_acc += (outputs.max(1)[1] == labels).sum().item()
            if output_hold.shape < torch.Size([1]):
                output_hold = outputs.max(1)[1]
                label_hold = labels
            else:
                output_hold = torch.cat([output_hold.cpu().detach(), outputs.max(1)[1].cpu().detach()], dim=0)
                label_hold = torch.cat([label_hold.cpu().detach(), labels.cpu().detach()], dim=0)
        avg_test_acc = test_acc / len(test_loader.dataset)
        print(f'test_acc: {avg_test_acc:.4f}')

    # make_confusion_matrix(f'./result/png/prototype_{prototype}/prune_finetune/abs/prune{k}_confusion_matrix.png',
    #                       f'./result/png/prototype_{prototype}/prune_finetune/abs/prune{k}_classification_report.png',
    #                       label_hold, output_hold)
    make_confusion_matrix(f'./result/png/fashionmnist_prototype{prototype}/prune_finetune/abs/prune{k}_confusion_matrix.png',
                          f'./result/png/fashionmnist_prototype{prototype}/prune_finetune/abs/prune{k}_classification_report.png',
                          label_hold, output_hold)
    # make_confusion_matrix(f'./result/png/prototype_{prototype}/prune_proto/prune_finetune/prune{k}_confusion_matrix.png',
    #                       f'./result/png/prototype_{prototype}/prune_proto/prune_finetune/prune{k}_classification_report.png',
    #                       label_hold, output_hold)
    # if offset <= 750:
    #     make_confusion_matrix(f'./result/png/prototype_{prototype}/epoch{offset - 50 + 1}_prune_confusion_matrix.png',
    #                           f'./result/png/prototype_{prototype}/epoch{offset - 50 + 1}_prune_classification_report.png',
    #                           label_hold, output_hold)
    # elif 200 < offset <= 300:
    #     make_confusion_matrix(f'./result/png/prototype_{prototype}/epoch{offset - 10 + 1}_prune_confusion_matrix.png',
    #                           f'./result/png/prototype_{prototype}/epoch{offset - 10 + 1}_prune_classification_report.png',
    #                           label_hold, output_hold)
    # elif offset == 800:
    #     make_confusion_matrix(f'./result/png/prototype_{prototype}/epoch{offset - 50}_prune_confusion_matrix.png',
    #                           f'./result/png/prototype_{prototype}/epoch{offset - 50}_prune_classification_report.png',
    #                           label_hold, output_hold)

# test_acc_vis
# learning_history = {'test_acc': []}
# learning_history['test_acc'].append('0.908')
# for k in range(2, 11):
#     # train_net = parameter_use(f'./result/pkl/prototype_{prototype}/prune_dense/not_abs/prune_proto_finetune_from_small/'
#     #                           f'prune_train_model_epoch{k}_{prototype}.pkl')
#     train_net = parameter_use(f'./result/pkl/fashionmnist_prototype{prototype}/prune_dense/abs/prune_conv_finetune_from_abs_small/'
#                               f'prune_train_model_epoch{k}_{prototype}.pkl')
#     # train_net = parameter_use(f'./result/pkl/prototype_{prototype}/prune_proto/prune_classifier_finetune_from_abs_large_epoch30/'
#     #                           f'prune_train_model_epoch{k}_{prototype}.pkl')
#     # if offset <= 700:
#     #     train_net = parameter_use(f'./result/pkl/prototype_{prototype}/acc_once_epoch50/train_model_epoch{offset + 1}_{prototype}.pkl')
#     # elif offset == 750:
#     #     train_net = parameter_use(f'./result/pkl/prototype_{prototype}/acc_once_epoch50/train_model_epoch{offset}_{prototype}.pkl')
#     # if offset <= 750:
#     #     offset += 50
#     # elif offset > 150:
#     #     offset += 10
#     with torch.no_grad():
#         avg_test_acc, test_acc = 0, 0
#         for images, labels in test_loader:
#             images, labels = images.to(device), labels.to(device)
#             ae_output, prototype_distances, feature_vector_distances, outputs, softmax_output = train_net(images)
#             test_acc += (outputs.max(1)[1] == labels).sum().item()
#         avg_test_acc = test_acc / len(test_loader.dataset)
#         learning_history['test_acc'].append(avg_test_acc)
# # pruningtestacc_vis(f'./result/png/prototype_{prototype}/testacc_afterprune.png', 10, learning_history)
# pruningtestacc_vis(f'./result/png/fashionmnist_prototype{prototype}/testacc_afterprune.png', 10, learning_history)
# # pruningtestacc_vis(f'./result/png/prototype_{prototype}/testacc_afterprune.png', 16, learning_history)
