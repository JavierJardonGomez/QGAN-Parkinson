# lstm_fcn_univariate.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Sequential):
    """Conv1d → BatchNorm → ReLU → Dropout (0.2)"""
    def __init__(self, in_ch, out_ch, k, p):
        super().__init__(
            nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        nn.init.kaiming_normal_(self[0].weight, nonlinearity="relu")


class LSTM_FCN_Univariate(nn.Module):
    """
    Implementa el LSTM-FCN original para series temporales univariantes
    (Karim et al., 2017) con kernels 8-5-3 y GAP final.
    """
    def __init__(self, input_size, num_classes,
                 lstm_units=128, dropout_fc=0.5):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size,  # ← 80
                            hidden_size=lstm_units,
                            batch_first=True)

        # ─── Rama LSTM ──────────────────────────────────────────────
        self.lstm = nn.LSTM(
            input_size=1,               # univariado ⇒ 1 canal
            hidden_size=lstm_units,
            batch_first=True
        )

        # ─── Rama FCN ───────────────────────────────────────────────
        self.conv1 = ConvBlock(1, 128, k=8, p=4)
        self.conv2 = ConvBlock(128, 256, k=5, p=2)
        self.conv3 = ConvBlock(256, 128, k=3, p=1)
        self.gap   = nn.AdaptiveAvgPool1d(1)

        # ─── Clasificador ───────────────────────────────────────────
        self.dropout = nn.Dropout(dropout_fc)
        self.fc      = nn.Linear(lstm_units + 128, num_classes)

    def forward(self, x):
        """
        x : tensor (batch, seq_len)  ó  (batch, seq_len, 1)
            • seq_len = nº de muestras del audio (p. ej. 16 000)
        """
        if x.ndim == 2:                # (B, L)  →  (B, L, 1)
            x = x.unsqueeze(-1)

        # ----- LSTM branch -----
        lstm_out, _ = self.lstm(x)     # (B, L, 128)
        lstm_out = lstm_out[:, -1, :]  # último paso → (B, 128)

        # ----- FCN branch -----
        y = x.permute(0, 2, 1)         # (B, 1, L) para Conv1d
        y = self.conv1(y)
        y = self.conv2(y)
        y = self.conv3(y)
        y = self.gap(y).squeeze(-1)    # (B, 128)

        # ----- Concatenate & classify -----
        out = torch.cat([lstm_out, y], dim=1)
        out = self.dropout(out)
        return self.fc(out)
