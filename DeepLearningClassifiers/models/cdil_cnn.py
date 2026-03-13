import sys
import torch
import torch.nn as nn
import torchvision.ops
from torch.nn.utils import weight_norm

# TCN
class tcn(nn.Module):
    def __init__(self, tcn_size):
        super(tcn, self).__init__()
        self.tcn_size = tcn_size

    def forward(self, x):
        return x[:, :, :-self.tcn_size].contiguous()


# Deformable convolution layer
class DeformableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, dilation, pad_mode):
        super(DeformableConv2d, self).__init__()

        self.padding = (padding, 0)
        self.dilation = (dilation, 1)
        self.kernel_size = (kernel_size, 1)

        self.offset_conv = nn.Conv2d(in_channels, 2 * kernel_size, self.kernel_size, padding=self.padding,
                                     dilation=self.dilation, padding_mode=pad_mode, bias=True)
        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)

        self.modulator_conv = nn.Conv2d(in_channels, kernel_size, self.kernel_size, padding=self.padding,
                                        dilation=self.dilation, padding_mode=pad_mode, bias=True)
        nn.init.constant_(self.modulator_conv.weight, 0.)
        nn.init.constant_(self.modulator_conv.bias, 0.)

        self.regular_conv = nn.Conv2d(in_channels, out_channels, self.kernel_size, padding=self.padding,
                                      dilation=self.dilation, padding_mode=pad_mode, bias=False)

    def forward(self, x):
        h, w = x.shape[2:]
        max_offset = max(h, w) / 4.
        offset = self.offset_conv(x).clamp(-max_offset, max_offset)
        modulator = 2. * torch.sigmoid(self.modulator_conv(x))

        return torchvision.ops.deform_conv2d(x, offset, self.regular_conv.weight, self.regular_conv.bias,
                                             padding=self.padding, dilation=self.dilation, mask=modulator)


# Basic block
class Block(nn.Module):
    def __init__(self, model, c_in, c_out, ks, pad, dil, deformable):
        super(Block, self).__init__()
        self.model = model
        self.deform = deformable

        pad_mode = 'circular' if model == 'CDIL' else 'zeros'

        if deformable:
            self.conv = DeformableConv2d(c_in, c_out, ks, pad, dil, pad_mode)
        else:
            self.conv = weight_norm(nn.Conv1d(c_in, c_out, ks, padding=pad, dilation=dil, padding_mode=pad_mode))
            nn.init.normal_(self.conv.weight, 0, 0.01)
            nn.init.normal_(self.conv.bias, 0, 0.01)

        if model == 'TCN':
            self.tcn = nn.Sequential(self.conv, tcn(pad))

        self.res = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else None
        if self.res is not None:
            nn.init.normal_(self.res.weight, 0, 0.01)
            nn.init.normal_(self.res.bias, 0, 0.01)

        self.nonlinear = nn.ReLU()

    def forward(self, x):
        net = self.tcn if self.model == 'TCN' else self.conv

        if self.deform:
            x_2d = x.unsqueeze(-1)
            out = net(x_2d)
            res = x if self.res is None else self.res(x)
            y = self.nonlinear(out) + res.unsqueeze(-1)
            return y.squeeze(-1)
        else:
            out = net(x)
            res = x if self.res is None else self.res(x)
            return self.nonlinear(out + res)


# Convolutional part
class ConvPart(nn.Module):
    def __init__(self, model, dim_in, hidden_channels, ks, deformable, dynamic):
        super(ConvPart, self).__init__()
        layers = []
        num_layer = len(hidden_channels)
        begin = 1 if dynamic else 0

        for i in range(begin, num_layer):
            this_in = dim_in if i == 0 else hidden_channels[i - 1]
            this_out = hidden_channels[i]

            if model == 'CNN':
                this_dilation = 1
                this_padding = int((ks - 1) / 2)
            else:
                this_dilation = 2 ** i
                if model == 'TCN':
                    this_padding = this_dilation * (ks - 1)
                elif model in ['CDIL', 'DIL']:
                    this_padding = int(this_dilation * (ks - 1) / 2)
                else:
                    print('no this model.')
                    sys.exit()

            if i < (num_layer-3):
                layers += [Block(model, this_in, this_out, ks, this_padding, this_dilation, False)]
            else:
                layers += [Block(model, this_in, this_out, ks, this_padding, this_dilation, deformable)]

        self.conv_net = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv_net(x)


# Full model
class CONV(nn.Module):
    def __init__(self, task, model, input_size, output_size, num_channels, kernel_size,
                 deformable=False, dynamic=False, use_embed=False, char_vocab=None, fix_length=True):
        super(CONV, self).__init__()
        self.task = task
        self.model = model
        self.dynamic = dynamic
        self.use_embed = use_embed
        self.fix_length = fix_length

        if self.use_embed:
            self.embedding = nn.Embedding(char_vocab, input_size)

        self.conv = ConvPart(model, input_size, num_channels, kernel_size, deformable, dynamic)

        if task != 'retrieval_4000':
            self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, x, mask=None):
        if self.use_embed:
            x = self.embedding(x)
        if not self.dynamic:
            x = x.permute(0, 2, 1).to(dtype=torch.float)

        y_conv = self.conv(x)

        if self.model == 'TCN':
            if self.fix_length:
                y_class = y_conv[:, :, -1]
            else:
                P = mask.unsqueeze(1).expand(y_conv.size(0), y_conv.size(1)).unsqueeze(2)
                y_class = y_conv.gather(2, P).squeeze(2)
        else:
            y_class = y_conv.mean(dim=2)

        #print(f"y_class shape: {y_class.shape}")
        return y_class if self.task == 'retrieval_4000' else self.linear(y_class)
