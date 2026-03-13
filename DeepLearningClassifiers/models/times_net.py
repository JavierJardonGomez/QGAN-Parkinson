import torch
import torch.nn as nn
import torch.nn.functional as F
from types import SimpleNamespace  

def FFT_for_Period(x, k=2):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]

class Inception_Block_V1(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        super(Inception_Block_V1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        kernels = []
        for i in range(self.num_kernels):
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=2 * i + 1, padding=i))
        self.kernels = nn.ModuleList(kernels)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res_list = []
        for i in range(self.num_kernels):
            res_list.append(self.kernels[i](x))
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res

class TimesBlock(nn.Module):
    def __init__(self, configs):
        super(TimesBlock, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.k = configs.top_k
        # parameter-efficient design
        self.conv = nn.Sequential(
            Inception_Block_V1(configs.d_model, configs.d_ff,
                               num_kernels=configs.num_kernels),
            nn.GELU(),
            Inception_Block_V1(configs.d_ff, configs.d_model,
                               num_kernels=configs.num_kernels)
        )

    def forward(self, x):
        #print("DEBUG  TimesBlock  x.shape =", x.shape) 
        B, T, N = x.size()
        period_list, period_weight = FFT_for_Period(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]
            # padding
            if (self.seq_len + self.pred_len) % period != 0:
                length = (
                                 ((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = (self.seq_len + self.pred_len)
                out = x
            # reshape
            out = out.reshape(B, length // period, period,
                              N).permute(0, 3, 1, 2).contiguous()
            # 2D conv: from 1d Variation to 2d Variation
            out = self.conv(out)
            # reshape back
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :(self.seq_len + self.pred_len), :])
        res = torch.stack(res, dim=-1)
        # adaptive aggregation
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(
            1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        # residual connection
        res = res + x
        return res

class TimesNetClassifier(nn.Module):
    def __init__(self, seq_len: int, nb_classes: int, d_model: int = 64,
                 n_blocks: int = 3, patch_len: int = 13, device: str = "gpu"):
        super().__init__()
        self.proj = nn.Conv1d(1, d_model, kernel_size=patch_len, stride=patch_len)
        
        # ---------- CONFIG COMPLETO ----------
        cfg = SimpleNamespace(
            seq_len   = seq_len // patch_len,   # tras proyección
            pred_len  = 0,                      # no usamos forecasting
            patch_len = patch_len,
            d_model   = d_model,
            d_ff      = d_model * 4,            
            top_k     = 5,
            num_kernels = 6,
            dropout   = 0.1,
            device    = device,
        )
        # --------------------------------------
        #print("Config seq_len =", cfg.seq_len) 
        self.blocks = nn.Sequential(*[TimesBlock(cfg) for _ in range(n_blocks)])
        self.norm  = nn.BatchNorm1d(d_model)
        self.gap   = nn.AdaptiveAvgPool1d(1)
        self.fc    = nn.Linear(d_model, nb_classes)

    def forward(self, x):                    # x: (B,1,T)
        x = self.proj(x)                     # (B,D,T//P)
        x = x.permute(0, 2, 1)
        x = self.blocks(x)
        x = x.permute(0, 2, 1)
        x = self.norm(x)
        x = self.gap(x).squeeze(-1)          # (B,D)
        return F.log_softmax(self.fc(x), 1)