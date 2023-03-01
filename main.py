import argparse
import os

import torch
import torch.nn as nn
import numpy as np
import time
import math
import seaborn as sns
from dglgo.utils.early_stop import EarlyStopping
import pandas as pd

from utils import get_data, evaluate, train, load_adj, test
from model import Transformer
import warnings

warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Press the green button in the gu-tter to run the script.
if __name__ == '__main__':
    parser = argparse.ArgumentParser("Demo of argparse")

    parser.add_argument('--dataset', type=str, default="SMD")
    parser.add_argument('--group', type=str, default="1-1")
    parser.add_argument('--num_head', type=int, default=8)
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--input_window', type=int, default=100)
    parser.add_argument('--mode', type=str, default="train", help="[train, test]")
    parser.add_argument('--model_path', type=str, default="psm")

    args = parser.parse_args()
    dataset = args.dataset
    print(args)
    seed = 3047
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.multiprocessing.set_start_method('spawn')
    input_window = args.input_window
    output_window = input_window
    batch_size = args.batch_size
    patience = args.patience
    epochs = args.epoch
    num_head = args.num_head
    group = args.group
    mode = args.mode
    model_path = args.model_path
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    calculate_loss_over_all_values = True
    need_stand = True

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
        if group == "0-0":
            train_data_ = np.load("data/SMD/SMD_train.npy")
            val_data_ = np.load("data/SMD/SMD_test.npy")
            filename = "smd"
        else:
            train_data_ = np.load("data/SMD/machine-" + group + "_train.npy")
            val_data_ = np.load("data/SMD/machine-" + group + "_test.npy")
            filename = "smd" + group
            model_path = args.model_path

        feature_size = 38
    elif dataset == "MSDS":
        # SMAP
        train_data_ = np.load("data/MSDS/train.npy")
        val_data_ = np.load("data/MSDS/test.npy")
        val_data_ = val_data_[:len(val_data_)]
        feature_size = 10
        filename = "msds"
        # need_stand = False
    elif dataset == "WADI":
        # WADI
        train_data_ = np.load("data/WADI/train.npy")
        val_data_ = np.load("data/WADI/test.npy")

        # train_data_ = pd.read_csv("data/WADI/WADI_train.csv")
        # train_data_ = train_data_.fillna(method="bfill")
        # train_data_ = train_data_.to_numpy()
        # train_data_ = train_data_[:, :]
        # val_data_ = pd.read_csv("data/WADI/WADI_test.csv")
        # val_data_ = val_data_.fillna(method="bfill")
        # val_data_ = val_data_.to_numpy()
        # val_data_ = val_data_[:, 1:-1]

        feature_size = 127
        filename = "wadi"
        # need_stand = False
    elif dataset == "SWAT":
        # SWAT

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

    early_stopping = EarlyStopping(patience, checkpoint_path=model_path + ".pth")

    train_data, train_scaler = get_data(data_=train_data_, need_stand=need_stand)
    val_data, val_scaler = get_data(data_=val_data_)

    src = torch.from_numpy(train_data.copy())

    a = src.T
    similarity = torch.cosine_similarity(a.unsqueeze(1), a.unsqueeze(0), dim=-1).cuda().float()
    np.save("similarity.npy", similarity.detach().cpu().numpy())
    cos_adj = similarity

    adj = similarity
    print("load adj finished")
    model = Transformer(feature_size=feature_size, num_layers=5, input_window=input_window, adj=adj,
                        batch_size=batch_size, num_head=num_head).cuda()

    if mode == "train":
        best_model = model
        total_num = sum([param.nelement() for param in model.parameters()])
        print("Number of parameter: %d" % total_num)
        criterion = nn.MSELoss()
        lr = 0.0005
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.98)

        best_val_loss = float("inf")
        trainloss = np.empty(epochs)
        valloss = np.empty(epochs)
        for epoch in range(1, epochs + 1):
            epoch_start_time = time.time()
            train_loss = train(train_data=train_data, epoch=epoch, scheduler=scheduler, optimizer=optimizer,
                               criterion=criterion, model=model, batch_size=batch_size,
                               output_window=output_window, input_window=input_window, scaler=train_scaler,
                               calculate_loss_over_all_values=calculate_loss_over_all_values, cos_adj=cos_adj)
            val_loss = evaluate(model, val_data, calculate_loss_over_all_values, output_window, criterion,
                                batch_size=batch_size, input_window=input_window, scaler=val_scaler, epoch=epoch)
            trainloss[epochs - 1] = train_loss
            valloss[epochs - 1] = val_loss
            print(
                '| end of epoch {:3d} | time: {:5.2f}s | train loss {:5.5f} | valid loss {:5.5f} | valid ppl {:8.2f}'.format(
                    epoch, (time.time() - epoch_start_time), train_loss, val_loss, math.exp(val_loss)))
            if best_val_loss > val_loss:
                best_val_loss = val_loss
                best_model = model
            scheduler.step()
            early_stopping.step(2 - val_loss, model)
            if early_stopping.early_stop:
                print("Early Stopping")
                break
        print("best val loss: ", best_val_loss)
        print("val loss", valloss)
        print("train loss", trainloss)
    # elif mode == "test":
    #     state_dict = torch.load('./' + args.model_path + '.pth')
    #     model.load_state_dict(state_dict)
    #     test_labels = pd.read_csv("./data/PSM/test_label.csv").label.values[:]
    #     test(model, train_data, val_data, input_window, test_labels)
    else:
        print("the mode only can select from train and test")
