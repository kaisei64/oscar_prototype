import os
import sys

pardir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(pardir)
from dataset import *
from model import ProtoNet
import torch
import torch.optim as optim
from util_func import batch_elastic_transform, list_of_norms
from loss import *

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
    train_loss, train_acc, train_ae_error, train_error_1, train_error_2 = 0, 0, 0, 0, 0
    for i, (images, labels) in enumerate(train_loader):
        elastic_images = batch_elastic_transform(images.reshape(-1, 784),
                                                 sigma=sigma, alpha=alpha, height=28, width=28).reshape(-1, 1, 28, 28)
        elastic_images, labels = torch.tensor(elastic_images, dtype=dtype).to(device), labels.to(device)
        optimizer.zero_grad()
        ae_output, prototype_distances, feature_vector_distances, outputs, softmax_output = net(elastic_images)
        class_error = criterion(outputs, labels)
        train_ae_error = torch.mean(list_of_norms(ae_output - images))
        train_error_1 = torch.mean(torch.min(feature_vector_distances))
        train_error_2 = torch.mean(torch.min(prototype_distances))
        loss = prototype_loss(class_error, train_ae_error, train_error_1, train_error_2, error_1_flag=True, error_2_flag=True)
        train_loss += loss.item()
        train_acc += (outputs.max(1)[1] == labels).sum().item()
        loss.backward()
        optimizer.step()
    avg_train_loss, avg_train_acc = train_loss / len(train_loader.dataset), train_acc / len(train_loader.dataset)
    avg_train_ae_error = train_ae_error / len(train_loader.dataset)
    avg_train_error_1, avg_train_error_2 = train_error_1 / len(train_loader.dataset), train_error_2 / len(train_loader.dataset)

    # val
    net.eval()
    val_loss, val_acc, val_ae_error, val_error_1, val_error_2 = 0, 0, 0, 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            labels = labels.to(device)
            ae_output, prototype_distances, feature_vector_distances, outputs, softmax_output = net(images)
            class_error = criterion(outputs, labels)
            train_ae_error = torch.mean(list_of_norms(ae_output - images))
            train_error_1 = torch.mean(torch.min(feature_vector_distances))
            train_error_2 = torch.mean(torch.min(prototype_distances))
            loss = prototype_loss(class_error, train_ae_error, train_error_1, train_error_2, error_1_flag=True,
                                  error_2_flag=True)
            val_loss += loss.item()
            val_acc += (outputs.max(1)[1] == labels).sum().item()
    avg_val_loss, avg_val_acc = val_loss / len(test_loader.dataset), val_acc / len(test_loader.dataset)
    avg_val_ae_error = val_ae_error / len(test_loader.dataset)
    avg_val_error_1, avg_val_error_2 = val_error_1 / len(test_loader.dataset), val_error_2 / len(test_loader.dataset)

    print(f'epoch [{epoch + 1}/{num_epochs}], train_loss: {avg_train_loss:.4f}'
          f', train_acc: {avg_train_acc:.4f}, 'f'val_loss: {avg_val_loss:.4f}, val_acc: {avg_val_acc:.4f}')
