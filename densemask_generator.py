import numpy as np


class DenseMaskGenerator:
    def __init__(self):
        self.prune_ratio = 0
        self.mask = None
        self.bias_mask = None
        self.count = 0

    def generate_mask(self, x, prune_ratio):
        self.prune_ratio = prune_ratio
        sub_x = x.cpu().detach().numpy()
        self.mask = np.where(np.abs(sub_x) <= self.prune_ratio, 0, 1)
        return self.mask

    def neuron_number(self, x):
        self.count = 0
        for i, j in enumerate(x):
            if np.all(self.mask.T[i] == 0):
                self.count += 1
        return self.count
