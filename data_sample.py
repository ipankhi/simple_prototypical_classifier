import torch
import numpy as np

class CategoriesSampler():

    def __init__(self, label, chosen_classes, n_batch,  n_per):
        self.n_batch = n_batch
        self.chosen_classes = chosen_classes
        self.n_per = n_per

        label = np.array(label)
        #print(label)
        self.m_ind = []
        for i in range(max(label) + 1):
            ind = np.argwhere(label == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)

    def __len__(self):
        return self.n_batch
    
    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = []
            for c in self.chosen_classes:
                l = self.m_ind[c]
                pos = torch.randperm(len(l))[:self.n_per]
                batch.append(l[pos])
            batch = torch.stack(batch).t().reshape(-1)
            yield batch

