import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialTopK(nn.Module):
    """ Spatial top k pooling

    For each channel, take only the top k activations
    groups: number of blocks of top k. # channels % blocks = 0 
    topk: total number of activations retained, divided over groups
    frac: alternative to topk, fraction of channel activations to retain

    ***Extend this to blocks of activations
    """
    def __init__(self, topk=1, frac=None, groups=1):
        super(SpatialTopK, self).__init__()
        self.topk = int(topk/groups)
        self.frac = frac
        self.groups = groups


    def forward(self, x):
        # hardcode this incase the alternative is real slow with the for loop.
        if self.groups==1:
            if self.frac:
                self.topk = int(x.shape[1]*self.frac)
            zeros = torch.zeros(x.shape).type(x.type())
            topk_num, topk_idx = torch.topk(x, k=self.topk, dim=1)
            x = zeros.scatter_(dim=1, index=topk_idx, src=topk_num)

        else:
            group_size = int(x.shape[1]/2)
            for group in range(self.groups):
                curr_group = x[:, group*group_size: (group+1)*group_size, :, :]
                if self.frac:
                    self.topk = int(curr_group.shape[1]*self.frac)
                zeros = torch.zeros(curr_group.shape)
                topk_num, topk_idx = torch.topk(curr_group, k=self.topk, dim=1)
                curr_group = zeros.scatter_(dim=1, index=topk_idx, src=topk_num)
                x[:, group*group_size: (group+1)*group_size, :, :] = curr_group
        return x