import os
import sys
pardir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(pardir)
from dataset import *
from model import ProtoNet, class_num, prototype_num
from util_func import batch_elastic_transform, list_of_norms, result_save, parameter_save
from loss import *
import torch
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

learning_history = {'epoch': [], 'train_acc': [], 'train_total_loss': [], 'train_class_loss': [], 'train_ae_loss': [],
                    'train_error_1_loss': [], 'train_error_2_loss': [], 'val_acc': [], 'val_total_loss': [],
                    'val_class_loss': [], 'val_ae_loss': [], 'val_error_1_loss': [], 'val_error_2_loss': [],
                    'test_acc': []}

net = ProtoNet().to(device)
optimizer = optim.Adam(net.parameters(), lr=0.002)

# training parameters
num_epochs = 1500
test_display_step = 100
save_step = 50

# elastic deformation parameters
sigma = 4
alpha = 20

for epoch in range(num_epochs):
    # train
    net.train()
    train_loss, train_acc, train_class_error, train_ae_error, train_error_1, train_error_2 = 0, 0, 0, 0, 0, 0
    for i, (images, labels) in enumerate(train_loader):
        elastic_images = batch_elastic_transform(images.reshape(-1, 784),
                                                 sigma=sigma, alpha=alpha, height=28, width=28).reshape(-1, 1, 28, 28)
        elastic_images, labels = torch.tensor(elastic_images, dtype=dtype).to(device), labels.to(device)
        optimizer.zero_grad()
        ae_output, prototype_distances, feature_vector_distances, outputs, softmax_output = net(elastic_images)
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
    avg_train_loss, avg_train_acc = train_loss / len(train_loader.dataset), train_acc / len(train_loader.dataset)
    avg_train_class_error, avg_train_ae_error = train_class_error / len(train_loader.dataset), train_ae_error / len(train_loader.dataset)
    avg_train_error_1, avg_train_error_2 = train_error_1 / len(train_loader.dataset), train_error_2 / len(train_loader.dataset)

    # val
    net.eval()
    val_loss, val_acc, val_class_error, val_ae_error, val_error_1, val_error_2 = 0, 0, 0, 0, 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            ae_output, prototype_distances, feature_vector_distances, outputs, softmax_output = net(images)
            class_error = criterion(outputs, labels)
            val_class_error += criterion(outputs, labels)
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
        avg_val_class_error, avg_val_ae_error = val_class_error / len(val_loader.dataset), val_ae_error / len(val_loader.dataset)
        avg_val_error_1, avg_val_error_2 = val_error_1 / len(val_loader.dataset), val_error_2 / len(val_loader.dataset)

    print(f'epoch [{epoch + 1}/{num_epochs}], train_loss: {avg_train_loss:.4f}'
          f', train_acc: {avg_train_acc:.4f}, 'f'val_loss: {avg_val_loss:.4f}, val_acc: {avg_val_acc:.4f}')

    # test
    avg_test_acc, test_acc = 0, 0
    if epoch % test_display_step == 0 or epoch == num_epochs - 1:
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            ae_output, prototype_distances, feature_vector_distances, outputs, softmax_output = net(images)
            test_acc += (outputs.max(1)[1] == labels).sum().item()
        avg_test_acc = test_acc / len(test_loader.dataset)
        print(f'test_acc: {avg_test_acc:.4f}')

    # save the learning history
    learning_history['epoch'].append(epoch+1)
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
    result_save('./result/csv/train_history.csv', learning_history)

    # save model, prototype and ae_out
    if epoch % save_step == 0 or epoch == num_epochs - 1:
        with torch.no_grad():
            parameter_save(f'./result/pkl/train_model_{epoch+1}.pkl', net)

            prototype_imgs = net.decoder(
                net.prototype_feature_vectors.reshape(prototype_num, class_num, 2, 2)).cpu().numpy()
            n_cols = 5
            n_rows = prototype_num // n_cols + 1 if prototype_num % n_cols != 0 else prototype_num // n_cols
            g, b = plt.subplots(n_rows, n_cols, figsize=(n_cols, n_rows))
            for i in range(n_rows):
                for j in range(n_cols):
                    if i * n_cols + j < prototype_num:
                        b[i][j].imshow(prototype_imgs[i * n_cols + j].reshape(28, 28),
                                       cmap='gray',
                                       interpolation='none')
                        b[i][j].axis('off')
            plt.savefig(f'./result/png/prototype_{epoch+1}.png', transparent=True, bbox_inches='tight', pad_inches=0)
            plt.close()

            examples_to_show = 10
            examples = [train_dataset[i][0] for i in range(examples_to_show)]
            examples = torch.cat(examples).reshape(len(examples), *examples[0].shape).to(device)
            encode_decode = net.decoder(net.encoder(examples))
            f, a = plt.subplots(2, examples_to_show, figsize=(examples_to_show, 2))
            for i in range(examples_to_show):
                a[0][i].imshow(examples[i].cpu().numpy().reshape(28, 28),
                               cmap='gray',
                               interpolation='none')
                a[0][i].axis('off')
                a[1][i].imshow(encode_decode[i].cpu().numpy().reshape(28, 28),
                               cmap='gray',
                               interpolation='none')
                a[1][i].axis('off')
            plt.savefig(f'./result/png/ae_decode_{epoch+1}.png', transparent=True, bbox_inches='tight', pad_inches=0)
            plt.close()
