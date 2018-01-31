#from https://github.com/pytorch/tnt/blob/master/torchnet/dataset/shuffledataset.py

import torch
class ShuffleDataset(object):
    def __init__(self, dataset, size=None, replacement=False):
        if size and not replacement and size > len(dataset):
            raise ValueError('size cannot be larger than underlying dataset \
                    size when sampling without replacement')

        super(ShuffleDataset, self).__init__(dataset,
                                            lambda dataset, idx: self.perm[idx],
                                            size)
        self.replacement = replacement
        self.resample()

    def resample(self, seed=None):
        if seed is not None:
            gen = torch.manual_seed(seed)
        else:
            gen = torch.default_generator

        if self.replacement:
            self.perm = torch.LongTensor(len(self)).random_(
                    len(self.dataset), generator=gen)
        else:
            self.perm = torch.randperm(
                    len(self.dataset), generator=gen).narrow(0, 0, len(self))
