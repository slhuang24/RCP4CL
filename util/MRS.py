import random

import numpy as np
import torch
import torch.nn.functional as F



def select_mixregion_u(predU,filter_foreground,max_topk,block_size ,stride):

    predU=predU.softmax(dim=1)
    UnConfu = -1.0 * torch.sum(predU * torch.log(predU + 1e-6), dim=1, keepdim=True).squeeze()
    # UnConfu = -1.0 * predU.max(dim=1)[0] This direct uncertainty estimation method can also achieve good performance.
    pred = predU.argmax(dim=1)
    pred = torch.where((pred > 0), torch.ones_like(pred), torch.zeros_like(pred)).float()
    B, H, W = UnConfu.shape
    mask = torch.zeros((B, H, W))

    for i in range(len(block_size)):
        folded_conf = F.avg_pool2d(UnConfu, kernel_size=block_size[i], stride=stride[i], padding=0)
        folded_pred = F.avg_pool2d(pred, kernel_size=block_size[i], stride=stride[i], padding=0)

        for b in range(B):
            randomk= random.randint(0, max_topk)
            filtered_indices = torch.nonzero(folded_pred[b] >= filter_foreground[i], as_tuple=False)
            if len(filtered_indices) < randomk:
                mask[b, :, :] = 0.0
                continue
            conf_values = folded_conf[b, filtered_indices[:, 0], filtered_indices[:, 1]]
            sorted_indices_conf = torch.argsort(conf_values, descending=False)

            top_conf_indices = filtered_indices[sorted_indices_conf][:max_topk]

            random_indices = torch.randperm(top_conf_indices.size(0))[:randomk]
            random_top_conf_indices = top_conf_indices[random_indices]

            for index in random_top_conf_indices:
                row_index, col_index = index
                mask[b, row_index * stride[i]:row_index *stride[i]+block_size[i],
                col_index * stride[i]:col_index * stride[i]+block_size[i]] = 1

    return mask

def select_mixregion_l(predL,label,filter_foreground,max_topk,block_size ,stride):

    UnConfl = -1.0 * torch.sum(predL * torch.log(predL + 1e-6), dim=1, keepdim=True).squeeze()
    # UnConfl = -1.0 * predL.max(dim=1)[0]

    pred = torch.where((label > 0), torch.ones_like(label), torch.zeros_like(label)).float()
    B, H, W = UnConfl.shape

    mask = torch.zeros((B, H, W))
    for i in range(len(block_size)):
        folded_conf = F.avg_pool2d(UnConfl, kernel_size=block_size[i], stride=stride[i], padding=0)
        folded_pred = F.avg_pool2d(pred, kernel_size=block_size[i], stride=stride[i], padding=0)

        for b in range(B):
            randomk = random.randint(0, max_topk)
            filtered_indices = torch.nonzero(folded_pred[b] > filter_foreground[i], as_tuple=False)
            if len(filtered_indices) < randomk:
                mask[b, :, :] = 0.0
                continue
            conf_values = folded_conf[b, filtered_indices[:, 0], filtered_indices[:, 1]]
            sorted_indices_conf = torch.argsort(conf_values, descending=False)

            top_conf_indices = filtered_indices[sorted_indices_conf][:max_topk]
            random_indices = torch.randperm(top_conf_indices.size(0))[:randomk]
            random_top_conf_indices = top_conf_indices[random_indices]

            for index in random_top_conf_indices:
                row_index, col_index = index
                mask[b, row_index * stride[i]:row_index *stride[i]+block_size[i],
                col_index * stride[i]:col_index * stride[i]+block_size[i]] = 1
    return mask



