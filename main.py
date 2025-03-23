from data_provider.data_factory import data_provider
from utils.tools import *
from tqdm import tqdm
from models.DLinear import DLinear
from models.PatchTST import PatchTST
from models.GPT4TS import GPT4TS
from models.Autoformer import Autoformer
from models.TimesNet import TimesNet
from models.GraphWaveNet import GraphWaveNet
from models.MTGNN import MTGNN
from models.GLALLM import GLALLM
import torch
import torch.nn as nn
import os
import time
import warnings
import numpy as np
import argparse
import random

warnings.filterwarnings('ignore')

fix_seed = 2025
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)
os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1, 2, 3, 4, 5, 6'

parser = argparse.ArgumentParser(description='GLALLM')

parser.add_argument('--is_train', type=int, default=1)
parser.add_argument('--checkpoints', type=str, default='./checkpoints/')

parser.add_argument('--root_path', type=str, default='./dataset/SDWPF/')
parser.add_argument('--data_path', type=str, default='SDWPF.csv')
parser.add_argument('--adj_data_path', type=str, default='adj.npy')
parser.add_argument('--dataset', type=str, default='dswe1')
parser.add_argument('--freq', type=str, default='h')
parser.add_argument('--embed', type=str, default='timeF')
parser.add_argument('--pe_mode', type=str, default='sincos')
parser.add_argument('--percent', type=list, default=[0.7, 0.8])
parser.add_argument('--dropout', type=float, default=0.05)
parser.add_argument('--num_nodes', type=int, default=134)
parser.add_argument('--num_edges', type=int, default=15)
parser.add_argument('--local_embed_dim', type=int, default=64)
parser.add_argument('--global_embed_dim', type=int, default=768)
parser.add_argument('--fused_embed_dim', type=int, default=128)
parser.add_argument('--graph_layer_num', type=int, default=2)
parser.add_argument('--tem_layer_num', type=int, default=3)
parser.add_argument('--memory_size', type=int, default=64)
parser.add_argument('--patch_size', type=int, default=12)
parser.add_argument('--stride', type=int, default=12)
parser.add_argument('--gpt_layers', type=int, default=2)

parser.add_argument('--seq_len', type=int, default=96)
parser.add_argument('--pred_len', type=int, default=12)
parser.add_argument('--label_len', type=int, default=1)

parser.add_argument('--model', type=str, default='GLALLM')

parser.add_argument('--decay_fac', type=float, default=0.75)
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--loss_func', type=str, default='mae')
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--num_workers', type=int, default=0)

parser.add_argument('--train_epochs', type=int, default=200)
parser.add_argument('--itr', type=int, default=100)
parser.add_argument('--lradj', type=str, default='cos')
parser.add_argument('--patience', type=int, default=10)
parser.add_argument('--tmax', type=int, default=10)
parser.add_argument('--device', type=int, default='0')


args = parser.parse_args()

if args.is_train:
    for ii in range(args.itr):

        setting = 'dataset_{}_model_{}_seq_len_{}_itr_{}'.format(args.dataset,
                                                          args.model,
                                                          args.seq_len,
                                                          ii)
        path = os.path.join(args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        train_data, train_loader = data_provider(args, 'train')
        vali_data, vali_loader = data_provider(args, 'val')
        test_data, test_loader = data_provider(args, 'test')

        device = torch.device('cuda:0')


        if args.model == 'DLinear':
            model = DLinear(args)
        elif args.model == 'PatchTST':
            model = PatchTST(args)
        elif args.model == 'GPT4TS':
            model = GPT4TS(args, device)
        elif args.model == 'Autoformer':
            model = Autoformer(args)
        elif args.model == 'TimesNet':
            model = TimesNet(args)
        elif args.model == 'GraphWaveNet':
            model = GraphWaveNet(args, device)
        elif args.model == 'MTGNN':
            model = MTGNN(args, device)
        elif args.model == 'GLALLM':
            model = GLALLM(args)

        model = nn.DataParallel(model, device_ids=[0, 1, 2, 3, 4, 5, 6]).to(device)

        params = model.parameters()
        model_optim = torch.optim.Adam(params, lr=args.learning_rate)

        early_stopping = EarlyStopping(patience=args.patience, verbose=True)

        if args.loss_func == 'mse':
            criterion = nn.MSELoss()
        elif args.loss_func == 'mae':
            criterion = nn.L1Loss()
        elif args.loss_func == 'smooth_mae':
            criterion = nn.SmoothL1Loss()

        if args.lradj == 'cos':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=args.tmax, eta_min=1e-8)
        elif args.lradj == 'pla':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(model_optim, mode='min', factor=0.5, patience=3,
                                                                   verbose=True, min_lr=1e-5)

        for epoch in range(args.train_epochs):
            model.train()
            iter_count = 0
            train_loss = []
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(train_loader)):

                model_optim.zero_grad()

                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)
                batch_x_mark = batch_x_mark.float().to(device)
                batch_y_mark = batch_y_mark.float().to(device)

                outputs = model(batch_x, batch_x_mark)

                loss = criterion(outputs, batch_y[:, -args.pred_len:, :])
                train_loss.append(loss.item())

                loss.backward()
                model_optim.step()

            train_loss = np.average(train_loss)
            valid_loss = valid(model, vali_loader, criterion, device, args)
            print("Epoch: {0}| Train Loss: {1:.7f} Valid Loss: {2:.7f}".format(
                epoch + 1, train_loss, valid_loss))

            scheduler.step(valid_loss)
            print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))

            early_stopping(valid_loss, model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        best_model_path = path + '/' + 'checkpoint.pth'
        model.load_state_dict(torch.load(best_model_path))
        print("------------------------------------")
        mse, mae = test(model, test_data, test_loader, device, args)

else:
    setting = 'dataset_{}_model_{}_seq_len_{}_itr_{}'.format(args.dataset,
                                                          args.model,
                                                          args.seq_len,
                                                          0)
    path = os.path.join(args.checkpoints, setting)
    test_data, test_loader = data_provider(args, 'test')

    device = torch.device('cuda:0')

    if args.model == 'DLinear':
        model = DLinear(args)
    elif args.model == 'PatchTST':
        model = PatchTST(args)
    elif args.model == 'GPT4TS':
        model = GPT4TS(args, device)
    elif args.model == 'Autoformer':
        model = Autoformer(args)
    elif args.model == 'TimesNet':
        model = TimesNet(args)
    elif args.model == 'GraphWaveNet':
        model = GraphWaveNet(args, device)
    elif args.model == 'MTGNN':
        model = MTGNN(args, device)
    elif args.model == 'GLALLM':
        model = GLALLM(args)

    model = nn.DataParallel(model, device_ids=[0, 1, 2, 3, 4, 5, 6]).to(device)
    best_model_path = path + '/' + 'checkpoint.pth'
    model.load_state_dict(torch.load(best_model_path))
    print("------------------------------------")
    mse, mae = test(model, test_data, test_loader, device, args)











