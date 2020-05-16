import torch.nn.functional as F
from dataset import *
from util_func import list_of_distances

class_num = 10
prototype_num = 15


class ProtoNet(nn.Module):

    def __init__(self):
        super(ProtoNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, class_num, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(class_num, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1),
            nn.Sigmoid()
        )
        self.prototype_feature_vectors = nn.Parameter(torch.rand((prototype_num, 2*2*class_num), device=device, dtype=dtype))
        self.classifier = nn.Sequential(
            nn.Linear(prototype_num, class_num, bias=False)
        )

    def forward(self, x):
        feature_vectors = self.encoder(x)
        ae_output = self.avgpool(feature_vectors)
        feature_vectors = torch.flatten(feature_vectors)
        prototype_distances = list_of_distances(feature_vectors, self.prototype_feature_vectors)
        feature_vector_distances = list_of_distances(self.prototype_feature_vectors, feature_vectors)
        output = self.classifier(feature_vectors)
        return ae_output, prototype_distances, feature_vector_distances, F.softmax(output), output
