import torch
import torch.nn.functional as F

from kymatio.scattering2d.frontend.base_frontend import ScatteringBase2D
from kymatio.frontend.torch_frontend import ScatteringTorch


class ScatteringTorch2D_wph(ScatteringTorch, ScatteringBase2D):
    def __init__(self, J, shape, L=8, max_order=1, pre_pad=False, backend='torch'):
        ScatteringTorch.__init__(self)
        ScatteringBase2D.__init__(**locals())
        ScatteringBase2D._instantiate_backend(self, 'kymatio.scattering2d.backend.')
        ScatteringBase2D.build(self)
        ScatteringBase2D.create_filters(self)

        self.register_filters()

    def register_filters(self):
        """ This function run the filterbank function that
            will create the filters as numpy array, and then, it
            saves those arrays as module's buffers."""
        # Create the filters

        self.register_buffer('phis', torch.from_numpy(self.phi[0]).unsqueeze(-1))

        psis = []
        for j in range(len(self.psi)):
            psis.append(torch.from_numpy(self.psi[j][0]).unsqueeze(-1))

        self.register_buffer('psis', torch.stack(psis, dim=0))

    def scattering(self, input):
        """Forward pass of the scattering.

            Parameters
            ----------
            input : tensor
                Tensor with k+2 dimensions :math:`(n_1, ..., n_k, M, N)` where :math:`(n_1, ...,n_k)` is
                arbitrary. Currently, k=2 is hardcoded. :math:`n_1` typically is the batch size, whereas
                :math:`n_2` is the number of input channels.

            Raises
            ------
            RuntimeError
                In the event that the input does not have at least two
                dimensions, or the tensor is not contiguous, or the tensor is
                not of the correct spatial size, padded or not.
            TypeError
                In the event that the input is not a Torch tensor.

            Returns
            -------
            S : tensor
                Scattering of the input, a tensor with k+3 dimensions :math:`(n_1, ...,n_k, D, Md, Nd)`
                where :math:`D` corresponds to a new channel dimension and :math:`(Md, Nd)` are
                downsampled sizes by a factor :math:`2^J`. Currently, k=2 is hardcoded.

        """
        if not torch.is_tensor(input):
            raise TypeError('The input should be a PyTorch Tensor.')

        if len(input.shape) < 2:
            raise RuntimeError('Input tensor must have at least two dimensions.')

        if not input.is_contiguous():
            raise RuntimeError('Tensor must be contiguous.')

        if (input.shape[-1] != self.N or input.shape[-2] != self.M) and not self.pre_pad:
            raise RuntimeError('Tensor must be of spatial size (%i,%i).' % (self.M, self.N))

        if (input.shape[-1] != self.N_padded or input.shape[-2] != self.M_padded) and self.pre_pad:
            raise RuntimeError('Padded tensor must be of spatial size (%i,%i).' % (self.M_padded, self.N_padded))

        batch_shape = input.shape[:-2]
        signal_shape = input.shape[-2:]
        input = input.reshape((-1,) + signal_shape)

        S = phase_scattering2d(input, self.pad, self.unpad, self.backend, self.J, self.phis, self.psis)
        scattering_shape = S.shape[-3:]
        S = S.reshape(batch_shape + scattering_shape)

        return S


def phase_scattering2d(x, pad, unpad, backend, J, phi, psi):
    subsample_fourier = backend.subsample_fourier
    fft = backend.fft
    cdgmm = backend.cdgmm

    out_S = []

    # x is of size (B, C, M_ori, N_ori)

    U_r = pad(x).unsqueeze(1)  # size (BC, 1, M, N, 2)
    U_0_c = fft(U_r, 'C2C')

    # First low pass filter
    U_1_c = cdgmm(U_0_c, phi)  # (BC, 1, M, N, 2)
    U_1_c = subsample_fourier(U_1_c, k=2 ** J)  # (BC, 1, M//2, N//2, 2)
    S_0 = fft(U_1_c, 'C2R', inverse=True)  # (BC, 1, M//2, N//2)
    S_0 = unpad(S_0)  # (BC, 1, M//2-2, N//2-2)
    out_S.append(S_0)
    U_1_c = cdgmm(U_0_c, psi)  # (BC, L, M, N, 2)
    U_1_c = subsample_fourier(U_1_c, k=2 ** J)  # (BC, L, M//2, N//2, 2)
    U_1_c = fft(U_1_c, 'C2C', inverse=True)  # (BC, L, M', N', 2) with M', N' either M, N or M//2, N//2
    S_1 = F.relu(torch.cat([U_1_c, -U_1_c], dim=-1))  # (BC, L, M', N', A) with A=4
    S_1 = S_1.permute(0, 1, -1, -3, -2).flatten(1, 2)  # (BC, AL, M', N') with A=4
    S_1 = unpad(S_1)  # (BC, AL, M//2-2, N//2-2)
    out_S.append(S_1)  # (BC, L, M//2-2, N//2-2)

    out_S = torch.cat(out_S, dim=1)
    return out_S


__all__ = ['ScatteringTorch2D_wph']
