import os
import sys
pardir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(pardir)
from util_func import parameter_use, make_confusion_matrix
from dataset import *
import torch

prototype = "15"
# training parameters
num_epochs = 5
save_step = 10
# elastic deformation parameters
sigma = 4
alpha = 20

for k in range(1, 11):
    train_net = parameter_use(f'./result/pkl/prototype_{prototype}/prune_dense/not_abs/prune_all_finetune_from_small/'
                              f'prune_train_model_epoch{k}_{prototype}.pkl')
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

    make_confusion_matrix(f'./result/png/prototype_{prototype}/prune_finetune/not_abs/prune{k}_confusion_matrix.png',
                          f'./result/png/prototype_{prototype}/prune_finetune/not_abs/prune{k}_classification_report.png',
                          label_hold, output_hold)
