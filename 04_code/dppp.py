import os
import numpy as np
import pandas as pd
import cvxpy as cp
from scipy.optimize import minimize
from joblib import Parallel, delayed

import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt

import torch
from torch import nn

output_file         = '05_output/'
osap_data           = '03_data/osap/'
temp_dir            = "D:/github_repositories"

os.environ["TEMP"]                  = temp_dir
os.environ["TMP"]                   = temp_dir
os.environ["JOBLIB_TEMP_FOLDER"]    = temp_dir
os.environ['PYTHONWARNINGS']        = 'ignore'

class portfolio_dataset(Dataset):
    """
    X: (n_stocks, n_time, n_features)
    y: (n_stocks, n_time, 1)  或 (n_stocks, n_time)

    返回:
        x_t: (n_stocks, n_features)
        y_t: (n_stocks, 1)
    """
    def __init__(self, X, y, dtype=torch.float32):
        X = torch.as_tensor(X, dtype=dtype)
        y = torch.as_tensor(y, dtype=dtype)

        if y.ndim == 2:
            y = y.unsqueeze(-1)

        assert X.ndim == 3, "X must be (stocks, time, features)"
        assert y.ndim == 3, "y must be (stocks, time, 1)"
        assert X.shape[0] == y.shape[0], "stock dimension mismatch"
        assert X.shape[1] == y.shape[1], "time dimension mismatch"

        self.X = X.permute(1, 0, 2).contiguous()
        self.y = y.permute(1, 0, 2).contiguous()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def create_dataloader(X, y, batch_size=60, shuffle=False, num_workers=0, drop_last=False):
    dataset = portfolio_dataset(X, y)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,   # 时间序列通常不要打乱
        num_workers=num_workers,
        drop_last=drop_last,
    )
    return dataset, loader

class cross_sectional_norm(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        mean                = x.mean(dim=1, keepdim=True)
        std                 = x.std(dim=1,  keepdim=True)
        return (x-mean)/std

class deep_ppp_nn(nn.Module):
    def __init__(self, input_dim, hidden_dim=(32, 16, 8), dropout=0, neg_slope=0, max_weight=0.03):
        super().__init__()
        self.max_weight     = max_weight
        prev_dim            = input_dim
        self.layers         = nn.ModuleList()
        
        self.layers.append(nn.Dropout(dropout))
        for h in hidden_dim:
            self.layers.append(nn.Linear(prev_dim, h))
            self.layers.append(nn.LeakyReLU(neg_slope))
            self.layers.append(nn.Dropout(dropout))
            # self.layers.append(cross_sectional_norm())
            prev_dim = h
        self.layers.append(nn.Linear(prev_dim, 1))

        self._init_weight()

    def _init_weight(self):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        T, N, L = x.shape
        
        # valid: (T, N, 1) — 该股票在该时间步有有效信号
        valid = ~torch.isnan(x).all(dim=-1, keepdim=True)
        Nt = valid.sum(dim=1, keepdim=True).float().clamp(min=1)  # (T, 1, 1)
        
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        tilt = x
        for layer in self.layers:
            tilt = layer(tilt)
        # tilt: (T, N, 1)

        # ---------- 只在 valid 股票上做截面标准化 ----------
        tilt_masked = tilt * valid                                          # invalid 位置置 0
        tilt_sum    = tilt_masked.sum(dim=1, keepdim=True)                 # (T, 1, 1)
        tilt_mean   = tilt_sum / Nt                                         # (T, 1, 1)
        tilt_demean = (tilt - tilt_mean) * valid                           # invalid 位置置 0
        tilt_var    = (tilt_demean ** 2).sum(dim=1, keepdim=True) / Nt
        tilt_std    = torch.sqrt(tilt_var + 1e-8)                          # 防止除 0
        tilt_norm   = tilt_demean / tilt_std                                # (T, N, 1)

        # ---------- 转换为权重：等权基础 + tilt ----------
        # tilt_norm 已截面零均值，除以 Nt 后加等权 1/Nt，保证权重截面和为 1
        w = (1.0 + tilt_norm) / Nt                                         # (T, N, 1)
        w = w * valid                                                       # invalid 股票权重为 0

        return w

    # def forward(self, x):
    #     # print('x is: ', x)
    #     T, N, L             = x.shape
    #     valid               = ~torch.isnan(x).all(dim=-1, keepdim=True)
    #     Nt                  = torch.sum(valid, dim=1, keepdim=True).expand(T, N, 1)
    #     x                   = torch.nan_to_num(x, nan=0, posinf=0, neginf=0)

    #     tilt = x
    #     for layer in self.layers:
    #         tilt = layer(tilt)
            
    #     # tilt                = torch.where(valid, tilt, torch.tensor(float('nan'), device=x.device))

    #     # normalize tilt
    #     tilt                = tilt*valid
    #     tilt_mean           = tilt.sum(dim=1, keepdim=True)/Nt
    #     tilt_demean         = tilt - tilt_mean
    #     tilt_std            = torch.sqrt((tilt_demean**2).sum(dim=1, keepdim=True)/Nt)
    #     tilt                = (tilt - tilt_mean)/(tilt_std + 1e-9)

    #     # calculate weight
    #     tilt                = tilt/Nt
    #     w                   = 1/Nt+tilt
    #     w                   = w*valid
    #     w                   = torch.where(valid, w, 0)

    #     return w
    
class crra_utility(nn.Module):
    def __init__(self, gamma = 5):
        super().__init__()
        self.gamma = gamma
    def forward(self, portfolio_ret):
        if np.abs(self.gamma - 1) <= 1e-6:
            utility = torch.log(1+portfolio_ret)
        else:
            utility = (1+portfolio_ret)**(1-self.gamma)/(1-self.gamma)

        loss = -utility.mean()
        return loss
    
class deep_ppp_trainer:
    def __init__(
        self, model, 
        device, 
        learning_rate = 0.01, 
        l1_penalty = 0, 
        gamma = 5, 
        utility_type = 'crra', 
        max_leverage = 1, 
        weight_constraint = 0.03,
    ):
        self.device             = device
        self.model              = model.to(self.device)
        self.learning_rate      = learning_rate
        self.l1_penalty         = l1_penalty
        self.gamma              = gamma
        self.utility_type       = utility_type
        self.max_leverage       = max_leverage
        self.weight_constraint  = weight_constraint
        self.patience_counter   = 0

        if self.utility_type == 'crra':
            self.utility_loss = crra_utility(self.gamma)

        self.utility_loss = self.utility_loss.to(self.device)

        self.optimizer          = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.best_val_loss    = float('inf')
        self.best_model_weights = None

    def compute_portfolio_return(self, weights, ret):
        T, N, _ = ret.shape
        ret_median = ret.nanmedian(dim=1, keepdim=True).values
        # print(ret_median.expand(T, N, 1))
        
        ret = torch.where(ret == ret, ret, ret_median)
        ret = torch.nan_to_num(ret, nan=0, posinf=0, neginf=0)
        portfolio_return = (weights*ret).sum(dim=1).squeeze(-1)
        return portfolio_return
    
    def compute_leverage_penalty(self, weights):
        leverage = torch.relu(-weights).sum(dim=1).mean()
        leverage_loss = torch.relu(leverage - self.max_leverage)
        return leverage_loss
    
    def compute_l1_penalty(self,):
        l1_loss = 0
        for param in self.model.parameters():
            l1_loss = l1_loss + torch.abs(param).sum()
        return l1_loss
    
    def train_step(self, batch_x, batch_y):
        self.model.train()
        self.optimizer.zero_grad()

        batch_x = batch_x.to(self.device)
        batch_y = batch_y.to(self.device)

        weights = self.model(batch_x)
        portfolio_return = self.compute_portfolio_return(weights, batch_y)

        utility_loss = self.utility_loss(portfolio_return)
        l1_loss = self.compute_l1_penalty()
        leverage_loss = self.compute_leverage_penalty(weights)

        total_loss = utility_loss + self.l1_penalty*l1_loss + 0.1*leverage_loss

        total_loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)
        self.optimizer.step()

        return total_loss.item(), utility_loss.item()
    
    def val_step(self, batch_x, batch_y):
        self.model.eval()
        
        with torch.no_grad():
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            weights = self.model(batch_x)
            portfolio_return = self.compute_portfolio_return(weights, batch_y)

            val_loss = self.utility_loss(portfolio_return)

        return val_loss.item(), portfolio_return

    def train_epoch(self, train_loader, val_loader=None, patience = 30):
        train_losses = []
        val_losses = []

        for batch_x, batch_y in train_loader:
            train_loss, _ = self.train_step(batch_x, batch_y)
            train_losses.append(train_loss)
        
        avg_train_loss = np.mean(train_losses)

        if val_loader is not None:
            for batch_x, batch_y in val_loader:
                val_loss, _ = self.val_step(batch_x, batch_y)
                val_losses.append(val_loss)
            
            avg_val_loss = np.mean(val_losses)
            
            if avg_val_loss < self.best_val_loss:
                self.best_val_loss = avg_val_loss
                self.patience_counter = 0
                self.best_model_weights = {
                    k: v.clone() for k, v in self.model.state_dict().items()
                }
            else:
                self.patience_counter += 1
                if self.patience_counter >= patience:
                    self.model.load_state_dict(self.best_model_weights)
                    return train_losses, val_losses, True
        
        return train_losses, val_losses, False
    
    def fit(self, train_loader, val_loader=None, epochs=200, patience=30, verbose=10):
        all_train_losses = []
        all_val_losses = []
        
        for epoch in range(epochs):
            train_losses, val_losses, early_stop = self.train_epoch(
                train_loader, val_loader, patience
            )
            
            all_train_losses.append(np.mean(train_losses))
            all_val_losses.append(np.mean(val_losses))
            
            if (epoch + 1) % verbose == 0:
                avg_train = np.mean(train_losses)
                msg = f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train:.6f}"
                if val_loader is not None:
                    avg_val = np.mean(val_losses)
                    msg += f", Val Loss: {avg_val:.6f}"
                if verbose >= 5:
                    print(msg)
            
            if early_stop:
                break
        return all_train_losses, all_val_losses
    
def save_loss_plot(train_loss, val_loss, output_file):
    plt.style.use('ggplot')
    fig, ax1 = plt.subplots(figsize=(8, 5))

    # 绘制曲线
    lns1 = ax1.plot(train_loss, label='Train Loss', color='steelblue', lw=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Train Loss')

    ax2 = ax1.twinx()
    lns2 = ax2.plot(val_loss, label='Validation Loss', color='darkorange', lw=2)
    ax2.set_ylabel('Validation Loss')
    ax2.grid(False) 

    # 合并图例
    lns = lns1 + lns2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='upper right')

    plt.tight_layout()
    plt.savefig(output_file, dpi=300) 
    plt.close(fig)

import torch
import numpy as np
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
if __name__ == '__main__':
    set_seed(42)
    signalnames         = np.load(osap_data + 'signalnames.npy', allow_pickle=True)
    date                = np.load(osap_data + 'date.npy')
    ret                 = np.load(osap_data + 'ret.npy')
    signals             = np.load(osap_data + 'signals.npy')

    i=80
    train_ret           = ret[:, i:i+240]
    train_signals       = signals[:, i:i+240, :]
    val_ret             = ret[:, i+240:i+300]
    val_signals         = signals[:, i+240:i+300, :]
    _, train_loader     = create_dataloader(train_signals, train_ret)
    _, val_loader       = create_dataloader(val_signals, val_ret)

    model = deep_ppp_nn(
        input_dim=train_signals.shape[-1], 
        hidden_dim=(32, 16, 8), 
        dropout=0.1, 
        neg_slope=0.01, 
        max_weight=0.03
    )
    trainer = deep_ppp_trainer(
        model=model,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        learning_rate=0.001,
        l1_penalty=1e-3,
        gamma=5,
        utility_type='crra',
        max_leverage=1,
        weight_constraint=0.03,
    )
    train_loss, val_loss = trainer.fit(train_loader, val_loader, epochs=200, patience=30, verbose=10)
    save_loss_plot(train_loss, val_loss, output_file)

