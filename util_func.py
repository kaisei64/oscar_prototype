import torch
import numpy as np
import cloudpickle
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import itertools
from dataset import batch_size, verbose
import time


def list_of_distances(x, y):
    """
    Given a list of vectors, X = [x_1, ..., x_n], and another list of vectors,
    Y = [y_1, ... , y_m], we return a list of vectors
            [[d(x_1, y_1), d(x_1, y_2), ... , d(x_1, y_m)],
             ...
             [d(x_n, y_1), d(x_n, y_2), ... , d(x_n, y_m)]],
    where the distance metric used is the squared euclidean distance.
    The computation is achieved through a clever use of broadcasting.
    """
    xx = torch.reshape(list_of_norms(x), shape=(-1, 1))
    yy = torch.reshape(list_of_norms(y), shape=(1, -1))
    output = xx + yy - 2 * torch.matmul(x, torch.t(y))
    return output


def list_of_norms(x):
    """
    X is a list of vectors X = [x_1, ..., x_n], we return
        [d(x_1, x_1), d(x_2, x_2), ... , d(x_n, x_n)], where the distance
    function is the squared euclidean distance.
    """
    return torch.pow(x, 2).sum(dim=1)


def result_save(path, learning_history):
    df = pd.DataFrame.from_dict(learning_history)
    df.to_csv(path)


def parameter_save(path, param):
    with open(path, 'wb') as f:
        cloudpickle.dump(param, f)


def parameter_use(path):
    with open(path, 'rb') as f:
        return cloudpickle.load(f)


def loss_vis(path, epoch, history):
    plt.figure()
    plt.plot(range(1, epoch + 1), history['train_loss'], label='train_loss')
    plt.plot(range(1, epoch + 1), history['train_loss'], label='train_loss')
    plt.xlabel('epoch')
    plt.legend()
    plt.show()
    # plt.savefig(path)


def testacc_vis(path, epoch, history):
    plt.figure(figsize=(6, 4.5), tight_layout=True)
    history['test_acc'] = [float(val) for val in history['test_acc']]
    acc_list = history['test_acc']
    if '0.0000' in acc_list:
        acc_list.remove('0.0000')
    plt.plot(range(1, epoch + 1), acc_list)
    # plt.title('test accuracy')
    # plt.xlabel('epoch')
    plt.tick_params(labelsize=24)
    ylabels = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    plt.yticks(ylabels, ylabels)
    # plt.show()
    plt.savefig(path)


def pruningtestacc_vis(path, epoch, history):
    plt.figure(figsize=(6, 4.5), tight_layout=True)
    history['test_acc'] = [float(val) for val in history['test_acc']]
    plt.plot(range(0, epoch), history['test_acc'])
    # plt.xlabel('Number of weights pruned', fontsize=12)
    # plt.ylabel('Accuracy', fontsize=12)
    plt.tick_params(labelsize=24)
    ylabels = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    plt.yticks(ylabels, ylabels)
    # plt.show()
    plt.savefig(path)


def conv_vis(path, param, fl_num, ch_num=0):
    sns_map = sns.heatmap(param[fl_num, ch_num, :, :], vmin=0.0, vmax=1.0)
    sns_map.figure.savefig(path)
    sns_map.figure.clear()


def weight_distribution_vis(path, param):
    sns.set_style("darkgrid")
    sns_plot = sns.distplot(param, rug=True)
    sns_plot.figure.savefig(path)
    sns_plot.figure.clear()


def plot_confusion_matrix(cm, classes, output_file,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    # plt.title(title)
    # plt.colorbar()
    tick_marks = np.arange(len(classes))
    # plt.xticks(tick_marks, classes, rotation=45)
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    plt.tick_params(labelsize=12)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",
                 fontsize=12)

    plt.tight_layout()
    plt.ylabel('True label', fontsize=14)
    plt.xlabel('Predicted label', fontsize=14)
    plt.savefig(output_file, bbox_inches="tight")


def make_confusion_matrix(path1, path2, label, pred):
    title = f"overall accuracy:{str(accuracy_score(label.numpy().tolist(), pred.numpy().tolist()))}\n"
    plt.figure()
    plt.text(0.1, 0.03, str(classification_report(label.numpy().tolist(), pred.numpy().tolist())), size=12)
    plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
    plt.tick_params(bottom=False, left=False, right=False, top=False)
    plt.savefig(path2)
    cnf_matrix = confusion_matrix(label.numpy().tolist(), pred.numpy().tolist(), labels=[i for i in range(10)])
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=[i for i in range(10)], output_file=path1, title=title)


def outlier_2s(col):
    average = np.median(col)
    sd = np.std(col)
    outlier_min = average - sd * 1
    outlier_max = average + sd * 1
    return outlier_min, outlier_max


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def compute_features(dataloader, model, N):
    if verbose:
        print('Compute features')
    batch_time = AverageMeter()
    end = time.time()
    model.eval()
    # discard the label information in the dataloader
    for i, (input_tensor, _) in enumerate(dataloader):
        input_var = torch.autograd.Variable(input_tensor.cuda(), volatile=True)
        aux = model(input_var).reshape(-1, 40).data.cpu().numpy()

        if i == 0:
            features = np.zeros((N, aux.shape[1]), dtype='float32')

        aux = aux.astype('float32')
        if i < len(dataloader) - 1:
            features[i * batch_size: (i + 1) * batch_size] = aux
        else:
            # special treatment for final batch
            # print(i)
            # print(len(aux))
            # print(aux.shape)
            # print(features[i*batch_size:].shape)
            features[i * batch_size:] = aux

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if verbose and (i % 50) == 0:
            print('{0} / {1}\t'
                  'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})'
                  .format(i, len(dataloader), batch_time=batch_time))
    return features
