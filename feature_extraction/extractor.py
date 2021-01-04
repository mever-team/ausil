import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.nn.init as init
from operator import itemgetter

class featExtractor(nn.Module):

    def __init__(self,model,layer_name):
        super(featExtractor,self).__init__()
        for k, mod in model._modules.items():
            self.add_module(k,mod)
        self.featLayer = layer_name

    def forward(self, x):
        logits = x
        tensors = []
        for nm, module in self._modules.items():
            logits = module(logits)
            if nm in self.featLayer:
                rv = F.normalize(logits, p=2, dim=1)
                rv = F.max_pool2d(rv, kernel_size=rv.size()[2:])
                tensors.append(F.normalize(rv, p=2, dim=1))
        logits = torch.cat(tensors, 1).view(logits.shape[0], -1)
        return logits
