import numpy as np
import torch
from tqdm import tqdm
from utils.metrics import metric

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


def valid(model, vali_loader, criterion, device, args):
    total_loss = []
    model.eval()

    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(vali_loader)):
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)

            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)

            outputs = model(batch_x, batch_x_mark)

            pred = outputs.detach().cpu()
            true = batch_y[:, -args.pred_len:, :].detach().cpu()

            loss = criterion(pred, true)

            total_loss.append(loss)
    total_loss = np.average(total_loss)

    return total_loss


def test(model, test_data, test_loader, device, args):
    preds = []
    trues = []

    model.eval()
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(test_loader)):

            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)

            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)

            outputs = model(batch_x, batch_x_mark)

            pred = outputs.detach().cpu().numpy()
            true = batch_y[:, -args.pred_len:, :].detach().cpu().numpy()

            preds.append(pred)
            trues.append(true)

    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    preds = test_data.inverse_transform(preds.reshape(-1, preds.shape[-1]))
    trues = test_data.inverse_transform(trues.reshape(-1, trues.shape[-1]))

    preds = preds.reshape(-1, args.pred_len, preds.shape[-1])  # [B, T, N]
    # np.save('./pred8_nrel.npy', preds)
    preds = preds.transpose((0, 2, 1))

    trues = trues.reshape(-1, args.pred_len, trues.shape[-1])
    # np.save('./true8_nrel.npy', trues)
    trues = trues.transpose((0, 2, 1))
    print('test shape:', preds.shape, trues.shape)

    amae = []
    armse = []
    amape = []
    for i in range(args.pred_len):
        pred = preds[:, :, i]
        true = trues[:, :, i]
        mae, mse, rmse, mape, r2, cvrmse = metric(pred, true)
        log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}, Test R2: {:.4f}, Test CVRMSE: {:.4f}'
        print(log.format(i + 1, mae, rmse, mape, r2, cvrmse))
        amae.append(mae)
        armse.append(rmse)
        amape.append(mape)



    log = 'On average over {} horizons, Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}'
    print(log.format(args.pred_len, np.mean(amae), np.mean(armse), np.mean(amape)))

    return np.mean(armse), np.mean(amae)

def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std