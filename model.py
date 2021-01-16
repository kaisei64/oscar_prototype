import torch.nn.functional as F
from dataset import *
from util_func import list_of_distances
import numpy as np

class_num = 10
prototype_num = 15
in_channel_num = 1


class ProtoNet(nn.Module):

    def __init__(self, num_prototype=prototype_num):
        super(ProtoNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channel_num, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, class_num, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(class_num, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, in_channel_num, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
        # self.simple_encoder = nn.Sequential(
        #     nn.Linear(in_width * in_height, 128),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(128, class_num),
        #     nn.ReLU(inplace=True)
        # )
        # self.simple_decoder = nn.Sequential(
        #     nn.Linear(class_num, 128),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(128, in_width * in_height),
        #     nn.Sigmoid()
        # )
        # self.cifar_encoder = nn.Sequential(
        #     nn.Conv2d(in_channel_num, 32, 3, stride=1, padding=1),  # [batch, 32, 32, 32]
        #     nn.ReLU(inplace=True),
        #     nn.BatchNorm2d(32),
        #     nn.Conv2d(32, 32, 3, stride=2, padding=1),  # [batch, 32, 16, 16]
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(32, class_num, 3, stride=1, padding=1),  # [batch, 10, 16, 16]
        #     nn.ReLU(inplace=True),
        #     nn.BatchNorm2d(class_num)
        # )
        # self.cifar_decoder = nn.Sequential(
        #     nn.ConvTranspose2d(class_num, 32, 3, stride=2, padding=1, output_padding=1),  # [batch, 32, 32, 32]
        #     nn.ReLU(inplace=True),
        #     nn.ConvTranspose2d(32, 32, 3, stride=1, padding=1),  # [batch, 32, 32, 32]
        #     nn.ReLU(inplace=True),
        #     nn.BatchNorm2d(32),
        #     nn.ConvTranspose2d(32, in_channel_num, 3, stride=1, padding=1),  # [batch, 1, 32, 32]
        #     nn.Sigmoid(),
        # )
        self.prototype_feature_vectors = nn.Parameter(
            # torch.nn.init.uniform_(torch.empty(prototype_num, class_num)))  # simple_ae
            torch.nn.init.uniform_(torch.empty(num_prototype, 2 * 2 * class_num)))
            # torch.nn.init.uniform_(torch.empty(prototype_num, 16 * 16 * class_num)))  # cifar10
        self.classifier = nn.Sequential(
            nn.Linear(num_prototype, class_num, bias=False)
        )

    def forward(self, x):
        feature_vectors = self.encoder(x)
        # feature_vectors = self.cifar_encoder(x)
        # x = x.view(-1, in_width * in_height)
        # feature_vectors = self.simple_encoder(x)
        ae_output = self.decoder(feature_vectors)
        # ae_output = self.cifar_decoder(feature_vectors)
        # ae_output = self.simple_decoder(feature_vectors)
        feature_vectors = feature_vectors.reshape(batch_size, -1)
        prototype_distances = list_of_distances(feature_vectors, self.prototype_feature_vectors)
        feature_vector_distances = list_of_distances(self.prototype_feature_vectors, feature_vectors)
        proto_proto_distances = list_of_distances(self.prototype_feature_vectors, self.prototype_feature_vectors).cpu().detach().numpy()
        proto_proto_distances_triangular_matrix = torch.tensor(np.triu(proto_proto_distances, k=1)).flatten()
        delete_index = np.where(proto_proto_distances_triangular_matrix == 0)
        proto_proto_distances_triangular_matrix = torch.tensor(np.delete(proto_proto_distances_triangular_matrix, delete_index))
        proto_proto_distances = torch.ones(proto_proto_distances_triangular_matrix.shape) + torch.tensor(2) - proto_proto_distances_triangular_matrix
        proto_proto_distances = torch.max(proto_proto_distances, torch.zeros(proto_proto_distances.shape))
        proto_proto_distances = torch.pow(proto_proto_distances, 2).sum()
        output = self.classifier(prototype_distances)
        return ae_output, prototype_distances, feature_vector_distances, proto_proto_distances, output, F.softmax(output)
