import torch.nn as nn
import torch
import numpy as np


class LinearProj(nn.Module):
    def __init__(self, standardization, proj, P_kernel_size=1, frame_kernel_size=1, frame_stride=1):
        super(LinearProj, self).__init__()
        self.standardization = standardization
        self.proj = proj
        self.P_kernel_size = P_kernel_size
        self.frame_kernel_size = frame_kernel_size
        self.frame_stride = frame_stride

    def forward(self, x):
        output = self.standardization(x)
        if self.P_kernel_size > 1:
            output = nn.functional.pad(output, ((self.P_kernel_size-1)//2,) * 4, mode='reflect')
        output = self.proj(output)

        if self.frame_kernel_size > 1:
            # x is represented by a redundant tensor of patches of shape (B, Ck^2, M', N'), which we normalize.
            output = torch.nn.functional.unfold(output, kernel_size=self.frame_kernel_size, stride=self.frame_stride)
            output = torch.div(output, output.norm(p=2, dim=1, keepdim=True))  # (B, Ck^2, M'N')
            output = output.reshape(output.shape[0], output.shape[1],
                                    int(np.sqrt(output.shape[2])), int(np.sqrt(output.shape[2])))  # (B, Ck^2, M', N')
        else:
            # Equivalent to the above but faster.
            output = torch.div(output, output.norm(p=2, dim=1, keepdim=True))

        return output

