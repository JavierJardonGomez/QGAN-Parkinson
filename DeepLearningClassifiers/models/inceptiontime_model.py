# inception_time.py
import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_odd(k: int) -> int:
    """Devuelve k si es impar; si no, k−1 (evita kernels pares)."""
    return k if k % 2 else k - 1


class InceptionBlock(nn.Module):
    """
    Bloque Inception 1-D con cuello de botella y cuatro ramas (3 conv + 1 max-pool).
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        base_kernel_size: int = 39,
        bottleneck_channels: int = 32,
        use_bottleneck: bool = True,
    ):
        super().__init__()

        self.use_bottleneck = use_bottleneck and in_channels > 1
        if self.use_bottleneck:
            self.bottleneck = nn.Conv1d(
                in_channels, bottleneck_channels, kernel_size=1, bias=False
            )
            in_channels = bottleneck_channels

        # Tres escalas: k, k/2, k/4  (todas impares)
        kernel_sizes = [_make_odd(base_kernel_size // (2 ** i)) for i in range(3)]

        self.conv_list = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=k,
                    padding=k // 2,  # "same" porque k es impar
                    bias=False,
                )
                for k in kernel_sizes
            ]
        )

        self.maxpool_branch = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
        )

        self.bn = nn.BatchNorm1d(out_channels * 4)  # 3 conv + 1 pool = 4 ramas
        self.relu = nn.ReLU()

    def forward(self, x):
        x_in = self.bottleneck(x) if self.use_bottleneck else x
        out = torch.cat(
            [conv(x_in) for conv in self.conv_list] + [self.maxpool_branch(x_in)],
            dim=1,
        )
        return self.relu(self.bn(out))


class InceptionBlockResidual(nn.Module):
    """
    Bloque Inception con conexión residual "identity" o proyección 1×1.
    Se coloca cada tercer bloque, siguiendo el paper de Fawaz et al. (2020).
    """
    def __init__(self, in_channels: int, out_channels: int, **kwargs):
        super().__init__()
        self.inception = InceptionBlock(in_channels, out_channels, **kwargs)

        self.projection = (
            None
            if in_channels == out_channels * 4
            else nn.Sequential(
                nn.Conv1d(in_channels, out_channels * 4, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_channels * 4),
            )
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        shortcut = x if self.projection is None else self.projection(x)
        return self.relu(self.inception(x) + shortcut)


class InceptionTimeClassifier(nn.Module):
    """
    Implementación base de InceptionTime para clasificación de series temporales.
    Por defecto: profundidad 6, 32 filtros por rama y kernel raíz 39→19→9.
    """
    def __init__(
        self,
        input_channels: int,
        nb_classes: int,
        *,
        depth: int = 6,
        nb_filters: int = 32,
        base_kernel_size: int = 39,
        use_residual: bool = True,
        use_bottleneck: bool = True,
    ):
        super().__init__()

        self.blocks = nn.ModuleList()
        in_ch = input_channels

        for d in range(depth):
            Block = InceptionBlockResidual if use_residual and d % 3 == 2 else InceptionBlock
            block = Block(
                in_ch,
                nb_filters,
                base_kernel_size=base_kernel_size,
                bottleneck_channels=32,
                use_bottleneck=use_bottleneck,
            )
            self.blocks.append(block)
            in_ch = nb_filters * 4  # cada bloque concatena 4×out_channels

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(in_ch, nb_classes)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = self.gap(x).squeeze(-1)
        return F.log_softmax(self.fc(x), dim=1)


class InceptionTimeEnsemble(nn.Module):
    """
    Wrapper opcional: crea un ensemble de N modelos InceptionTime y
    promedia (o vota) las predicciones. Ajusta n_models según lo necesites.
    """
    def __init__(self, n_models: int, **kwargs):
        super().__init__()
        self.models = nn.ModuleList(
            [InceptionTimeClassifier(**kwargs) for _ in range(n_models)]
        )

    def forward(self, x):
        # Apila logits y promedia (equivalente a "soft-voting" en log-espacio)
        logits = torch.stack([m(x) for m in self.models], dim=0)
        return logits.mean(0)
