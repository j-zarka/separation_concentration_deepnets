import torch
import torch.nn as nn


def hook_fn(grad):
    if torch.isnan(grad).any():
        grad[torch.isnan(grad)] = 0.

    if torch.isinf(grad).any():
        grad[torch.isinf(grad)] = 0.

    return grad


class ScatNetAnalysis(nn.Module):
    def __init__(self, scattering, linear_proj, analysis, classifier):
        super(ScatNetAnalysis, self).__init__()
        self.scattering = scattering
        self.linear_proj = linear_proj
        self.analysis = analysis
        self.classifier = classifier

    def forward(self, x, j=0):
        if j > 0 and self.training:
            x.register_hook(hook_fn)

        output = self.scattering(x)
        output = self.linear_proj(output)
        output = self.analysis(output)
        output = self.classifier(output)

        return output
