import argparse
import os
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import time
import math
import torch.nn.functional as F
import pandas as pd20
from sklearn.metrics import ndcg_score
import seaborn as sns

import warnings

warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", category=DeprecationWarning)


def hit_att(ascore, labels, ps=[100, 150]):
    res = {}
    for p in ps:
        hit_score = []
        for i in range(ascore.shape[0]):
            a, l = ascore[i], labels[i]
            a, l = np.argsort(a).tolist()[::-1], set(np.where(l == 1)[0])
            if l:
                size = round(p * len(l) / 100)
                a_p = set(a[:size])
                intersect = a_p.intersection(l)
                hit = len(intersect) / len(l)
                hit_score.append(hit)
        res[f'Hit@{p}%'] = np.mean(hit_score)
    return res


def ndcg(ascore, labels, ps=[100, 150]):
    res = {}
    for p in ps:
        ndcg_scores = []
        for i in range(ascore.shape[0]):
            a, l = ascore[i], labels[i]
            labs = list(np.where(l == 1)[0])
            if labs:
                k_p = round(p * len(labs) / 100)
                try:
                    hit = ndcg_score(l.reshape(1, -1), a.reshape(1, -1), k=k_p)
                except Exception as e:
                    return {}
                ndcg_scores.append(hit)
        res[f'NDCG@{p}%'] = np.mean(ndcg_scores)
    return res


def kl(series_s, prior_s):
    items = F.kl_div(F.log_softmax(series_s), F.softmax(prior_s), reduce=False) \
            + F.kl_div(F.log_softmax(prior_s), F.softmax(series_s), reduce=False)
    # item = torch.mean(torch.mean(items, dim=-1), dim=-1)
    return items


# Press the green button in the gu-tter to run the script.
if __name__ == '__main__':
    parser = argparse.ArgumentParser("Demo of argparse")

    parser.add_argument('--dataset', type=str, default="SMD")
    parser.add_argument('--group', type=str, default="1-1")

    args = parser.parse_args()
    dataset = args.dataset
    print(args)

    if dataset == "MSL":
        # MSL
        train_data_ = np.load("data/MSL/MSL_train.npy")
        val_data_ = np.load("data/MSL/MSL_test.npy")

        # train_data_ = np.load("data/MSL/C-2_train.npy")
        # val_data_ = np.load("data/MSL/C-2_test.npy")
        feature_size = 55
        filename = "msllstm"
    elif dataset == "PSM":
        # PSM
        train_data_ = pd.read_csv("data/PSM/train.csv")
        train_data_ = train_data_.fillna(method="bfill")
        train_data_ = train_data_.to_numpy()
        train_data_ = train_data_[:, 1:]
        val_data_ = pd.read_csv("data/PSM/test.csv")
        val_data_ = val_data_.fillna(method="bfill")
        val_data_ = val_data_.to_numpy()
        val_data_ = val_data_[:, 1:]
        feature_size = 25
        label = pd.read_csv("data/PSM/test_label.csv").label.values
        filename = "psmlstm"
    elif dataset == "SMAP":
        # SMAP
        train_data_ = np.load("data/SMAP/SMAP_train.npy")
        val_data_ = np.load("data/SMAP/SMAP_test.npy")
        val_data_ = val_data_[:len(val_data_)]
        feature_size = 25
        filename = "smaplstm"
    elif dataset == "SMD":
        # SMD
        train_data_ = np.load("data/SMD/machine-" + args.group + "_train.npy")
        val_data_ = np.load("data/SMD/machine-" + args.group + "_test.npy")
        labels = np.load("./data/SMD/machine-" + args.group + "_labels.npy")
        label = np.load("./data/SMD/machine-" + args.group + "_label.npy")
        train = train_data_
        # args.group = "11"
        pre = np.load("./presmd" + args.group + "test_20.npy")
        tru = np.load("./truthsmd" + args.group + "test_20.npy")
        adj = torch.from_numpy(np.load("./adjsmd" + args.group + "test.npy"))

        feature_size = 38
        filename = "smd"
    elif dataset == "MSDS":
        # SMAP
        train_data_ = np.load("data/MSDS/train.npy")
        val_data_ = np.load("data/MSDS/test.npy")
        feature_size = 10
        filename = "msds"

        labels = np.load("./data/MSDS/labels.npy")
        pre = np.load("./premsdstest_20.npy")
        tru = np.load("./truthmsdstest_20.npy")
        label = np.load("./data/MSDS/label.npy")
        train = train_data_
        adj = torch.from_numpy(np.load("./adjmsdstest.npy"))
    elif dataset == "WADI":
        # WADI
        train_data_ = np.load("data/WADI/train.npy")
        val_data_ = np.load("data/WADI/test.npy")

        labels = np.load("./data/WADI/labels.npy")
        pre = np.load("./res/prewadi_20.npy")
        tru = np.load("./res/truthwadi_20.npy")
        label = np.load("./data/WADI/label.npy")
        train = train_data_
        adj = torch.from_numpy(np.load("./res/adjwadi.npy"))

        feature_size = 127
        filename = "wadi"
        # need_stand = False
    elif dataset == "SWAT":
        # SMAP
        train_data_ = np.load("data/WADI/train.npy")
        val_data_ = np.load("data/WADI/test.npy")

        train_data_ = pd.read_csv("data/SWAT/SWaT_train.csv")
        train_data_ = train_data_.fillna(method="bfill")
        train_data_ = train_data_.to_numpy()
        train_data_ = train_data_[:, 1:-1]
        val_data_ = pd.read_csv("data/SWAT/SWaT_test.csv")
        val_data_ = val_data_.fillna(method="bfill")
        val_data_ = val_data_.to_numpy()
        val_data_ = val_data_[:, 1:-2]

        feature_size = 51
        filename = "swat"
        # need_stand = False
    else:
        print("THE DATA IS NOT INCLUDE IN THIS PROJECT.")
        raise RuntimeError('THE DATA IS NOT INCLUDE IN THIS PROJECT.')

    mse = torch.nn.MSELoss(reduce=False)
    losss = mse(torch.from_numpy(pre), torch.from_numpy(tru))[:, 0, :]

    src = torch.from_numpy(train.astype(np.float32))
    a = src.T
    similarity = torch.cosine_similarity(a.unsqueeze(1), a.unsqueeze(0), dim=-1)
    a = similarity

    c = torch.zeros(adj.shape)
    for i in range(len(adj)):
        c[i] = kl(adj[i], a)
    c = torch.mean(c, dim=-1)

    # c = c.diff(dim=0)
    labels = labels[len(labels)-len(c):]
    losss = losss[len(losss)-len(c):]
    print(c.shape)
    print(labels.shape)
    # 1/0

    print(hit_att(c, labels))
    print(ndcg(c, labels))
    # print(hit_att(c * losss, labels))
    # print(ndcg(c * losss, labels))
    # print(hit_att(losss, labels))
    # print(ndcg(losss, labels))
