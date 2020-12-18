import torch.nn as nn
import torch
import numpy as np


class Analysis(nn.Module):
    def __init__(self, nb_channels_in, frame_width, lambda_, frame_kernel_size=1, frame_stride=1,
                 non_linearity='relu', n_space=None):

        super(Analysis, self).__init__()

        frame = nn.Conv2d(nb_channels_in, frame_width, kernel_size=frame_kernel_size, stride=1, padding=0,
                          bias=False).weight.data
        nn.init.orthogonal_(frame)

        input_size = nb_channels_in * frame_kernel_size ** 2
        self.trace_gram = min(input_size, frame_width)
        if frame_width > input_size:
            self.frame_norm_mean = np.sqrt(input_size / frame_width)
        else:  # controlled by Parseval regularization in this case
            self.frame_norm_mean = 1.
        frame.data *= self.frame_norm_mean / frame.data.norm(p=2, dim=(1, 2, 3), keepdim=True)

        self.frame_weight = nn.parameter.Parameter(frame)

        self.frame_width = frame_width
        self.frame_kernel_size = frame_kernel_size
        self.frame_stride = frame_stride

        self.register_buffer("lambda_", torch.FloatTensor(1).fill_(lambda_))
        self.non_linearity = non_linearity

        if self.frame_kernel_size > 1:
            shape = (1, nb_channels_in, n_space, n_space)
            ones = torch.ones(shape)
            fold_params = dict(stride=self.frame_stride, kernel_size=self.frame_kernel_size)
            patch_count = nn.functional.fold(nn.functional.unfold(ones, **fold_params),
                                             **fold_params, output_size=shape[2:])
            factor = 1 / patch_count
            self.register_buffer("factor", factor)

    def forward(self, x):
        with torch.no_grad():
            self.frame_weight.data *= self.frame_norm_mean / \
                                      self.frame_weight.data.norm(p=2, dim=(1, 2, 3), keepdim=True)

        if self.frame_kernel_size > 1:
            # In this case, x is represented by a redundant tensor of patches of shape (B, Ck^2, M', N')
            z = nn.functional.conv2d(x, self.frame_weight.reshape(self.frame_weight.shape[0], -1, 1, 1), stride=1)
        else:
            z = nn.functional.conv2d(x, self.frame_weight, stride=self.frame_stride)
        z = dispatch_non_linearity(self.non_linearity, z, self.lambda_)

        D_z = nn.functional.conv_transpose2d(z, self.frame_weight, stride=self.frame_stride)
        if self.frame_kernel_size > 1:
            # Average overlapping patches instead of summing them.
            D_z = D_z * self.factor

        return D_z


def dispatch_non_linearity(name, x, lambd):
    non_linearity = non_linearities[name]
    return non_linearity(x, lambd)


def relu(x, lambd):
    return nn.functional.relu(x - lambd)


def absolute(x, lambd):
    return torch.abs(x)


def softshrink(x, lambd):
    return nn.functional.relu(x - lambd) - nn.functional.relu(-x - lambd)


non_linearities = dict(absolute=absolute, relu=relu, softshrink=softshrink)
