import pywt
import math
import torch
import numpy as np
import torch.nn as nn
from torch.nn import Module
import torch.nn.functional as F
from torch.autograd import Function

class DWTFunction_1D(Function):
    @staticmethod
    def forward(ctx, input, matrix_Low, matrix_High):
        ctx.save_for_backward(matrix_Low, matrix_High)
        L = torch.matmul(input, matrix_Low.t())
        H = torch.matmul(input, matrix_High.t())
        return L, H

    @staticmethod
    def backward(ctx, grad_L, grad_H):
        matrix_L, matrix_H = ctx.saved_variables
        grad_input = torch.add(torch.matmul(
            grad_L, matrix_L), torch.matmul(grad_H, matrix_H))
        return grad_input, None, None


class IDWTFunction_1D(Function):
    @staticmethod
    def forward(ctx, input_L, input_H, matrix_L, matrix_H):
        ctx.save_for_backward(matrix_L, matrix_H)
        output = torch.add(torch.matmul(input_L, matrix_L),
                           torch.matmul(input_H, matrix_H))
        return output

    @staticmethod
    def backward(ctx, grad_output):
        matrix_L, matrix_H = ctx.saved_variables
        grad_L = torch.matmul(grad_output, matrix_L.t())
        grad_H = torch.matmul(grad_output, matrix_H.t())
        return grad_L, grad_H, None, None

class DWT_1D(Module):
    """
    input: the 1D data to be decomposed -- (N, C, Length)
    output: lfc -- (N, C, Length/2)
            hfc -- (N, C, Length/2)
    """

    def __init__(self, wavename):
        """
        1D discrete wavelet transform (DWT) for sequence decomposition
        """
        super(DWT_1D, self).__init__()
        wavelet = pywt.Wavelet(wavename)
        self.band_low = wavelet.rec_lo
        self.band_high = wavelet.rec_hi
        assert len(self.band_low) == len(self.band_high)
        self.band_length = len(self.band_low)
        assert self.band_length % 2 == 0
        self.band_length_half = math.floor(self.band_length / 2)

    def get_matrix(self):
        """
        generating the matrices: \mathcal{L}, \mathcal{H}
        :return: self.matrix_low = \mathcal{L}, self.matrix_high = \mathcal{H}
        """
        L1 = self.input_height
        L = math.floor(L1 / 2)
        matrix_h = np.zeros((L, L1 + self.band_length - 2))
        matrix_g = np.zeros((L1 - L, L1 + self.band_length - 2))
        end = None if self.band_length_half == 1 else (
            - self.band_length_half + 1)
        index = 0
        for i in range(L):
            for j in range(self.band_length):
                matrix_h[i, index + j] = self.band_low[j]
            index += 2
        index = 0
        for i in range(L1 - L):
            for j in range(self.band_length):
                matrix_g[i, index + j] = self.band_high[j]
            index += 2
        matrix_h = matrix_h[:, (self.band_length_half - 1):end]
        matrix_g = matrix_g[:, (self.band_length_half - 1):end]
        if torch.cuda.is_available():
            self.matrix_low = torch.Tensor(matrix_h).cuda()
            self.matrix_high = torch.Tensor(matrix_g).cuda()
        else:
            self.matrix_low = torch.Tensor(matrix_h)
            self.matrix_high = torch.Tensor(matrix_g)

    def forward(self, input):
        """
        input_low_frequency_component = \mathcal{L} * input
        input_high_frequency_component = \mathcal{H} * input
        :param input: the data to be decomposed
        :return: the low-frequency and high-frequency components of the input data
        """
        assert len(input.size()) == 3
        self.input_height = input.size()[-1]
        self.get_matrix()
        return DWTFunction_1D.apply(input, self.matrix_low, self.matrix_high)


class IDWT_1D(Module):
    """
    input:  lfc -- (N, C, Length/2)
            hfc -- (N, C, Length/2)
    output: the original data -- (N, C, Length)
    """

    def __init__(self, wavename):
        """
        1D inverse DWT (IDWT) for sequence reconstruction
        """
        super(IDWT_1D, self).__init__()
        wavelet = pywt.Wavelet(wavename)
        self.band_low = wavelet.dec_lo
        self.band_high = wavelet.dec_hi
        self.band_low.reverse()
        self.band_high.reverse()
        assert len(self.band_low) == len(self.band_high)
        self.band_length = len(self.band_low)
        assert self.band_length % 2 == 0
        self.band_length_half = math.floor(self.band_length / 2)

    def get_matrix(self):
        """
        generating the matrices: \mathcal{L}, \mathcal{H}
        :return: self.matrix_low = \mathcal{L}, self.matrix_high = \mathcal{H}
        """
        L1 = self.input_height
        L = math.floor(L1 / 2)
        matrix_h = np.zeros((L, L1 + self.band_length - 2))
        matrix_g = np.zeros((L1 - L, L1 + self.band_length - 2))
        end = None if self.band_length_half == 1 else (
            - self.band_length_half + 1)
        index = 0
        for i in range(L):
            for j in range(self.band_length):
                matrix_h[i, index + j] = self.band_low[j]
            index += 2
        index = 0
        for i in range(L1 - L):
            for j in range(self.band_length):
                matrix_g[i, index + j] = self.band_high[j]
            index += 2
        matrix_h = matrix_h[:, (self.band_length_half - 1):end]
        matrix_g = matrix_g[:, (self.band_length_half - 1):end]
        if torch.cuda.is_available():
            self.matrix_low = torch.Tensor(matrix_h).cuda()
            self.matrix_high = torch.Tensor(matrix_g).cuda()
        else:
            self.matrix_low = torch.Tensor(matrix_h)
            self.matrix_high = torch.Tensor(matrix_g)

    def forward(self, L, H):
        """
        :param L: the low-frequency component of the original data
        :param H: the high-frequency component of the original data
        :return: the original data
        """
        assert len(L.size()) == len(H.size()) == 3
        self.input_height = L.size()[-1] + H.size()[-1]
        self.get_matrix()
        return IDWTFunction_1D.apply(L, H, self.matrix_low, self.matrix_high)

class Conv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.weight)
        nn.init.zeros_(self.bias)


class PositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, noise_level):
        noise_level = noise_level.view(-1)
        count = self.dim // 2
        step = torch.arange(count, dtype=noise_level.dtype,
                            device=noise_level.device) / count
        encoding = noise_level.unsqueeze(
            1) * torch.exp(-ln(1e4) * step.unsqueeze(0))
        encoding = torch.cat(
            [torch.sin(encoding), torch.cos(encoding)], dim=-1)
        return encoding

class ResBlock(nn.Module):
    def __init__(self, input_size, hidden_size, dilation):
        super().__init__()

        self.filters = nn.ModuleList([
            Conv1d(input_size, hidden_size//4, 3, dilation=dilation, 
                   padding=1*dilation, padding_mode='reflect'),
            Conv1d(input_size, hidden_size//4, 5, dilation=dilation, 
                   padding=2*dilation, padding_mode='reflect'),
            Conv1d(input_size, hidden_size//4, 7, dilation=dilation, 
                   padding=3*dilation, padding_mode='reflect'),
            Conv1d(input_size, hidden_size//4, 9, dilation=dilation, 
                   padding=4*dilation, padding_mode='reflect'),
        ])

        self.conv_1 = Conv1d(hidden_size, hidden_size, 9, padding=4, 
                             padding_mode='reflect')

        self.norm = nn.InstanceNorm1d(hidden_size//2)

        self.conv_2 = Conv1d(hidden_size, hidden_size, 9, padding=4, 
                             padding_mode='reflect')

    def forward(self, x):
        residual = x

        filts = []
        for layer in self.filters:
            filts.append(layer(x))

        filts = torch.cat(filts, dim=1)

        nfilts, filts = self.conv_1(filts).chunk(2, dim=1)

        filts = F.leaky_relu(torch.cat([self.norm(nfilts), filts], dim=1), 0.2)

        filts = F.leaky_relu(self.conv_2(filts), 0.2)

        return filts + residual

class Projection(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.noise_func = nn.Linear(input_size, hidden_size)
        self.input_conv = Conv1d(input_size, input_size, 3, padding=1, 
                                 padding_mode='reflect')
        self.output_conv = Conv1d(input_size, hidden_size, 3, padding=1, 
                                  padding_mode='reflect')

    def forward(self, x, noise_embed):
        batch = x.shape[0]
        x = self.input_conv(x)
        x = x + self.noise_func(noise_embed).view(batch, -1, 1)
        return self.output_conv(x)

class WModel(nn.Module):
    def __init__(self, w_name="haar", feats=64):
        super(WModel, self).__init__()
        self.stream_x = nn.ModuleList([
            nn.Sequential(Conv1d(2, feats, 9, padding=4, padding_mode='reflect'),
                          nn.LeakyReLU(0.2)),
            ResBlock(feats, feats, 1),
            ResBlock(feats, feats, 2),
            ResBlock(feats, feats, 4),
            ResBlock(feats, feats, 2)
        ])

        self.stream_cond = nn.ModuleList([
            nn.Sequential(Conv1d(2, feats, 9, padding=4, padding_mode='reflect'),
                          nn.LeakyReLU(0.2)),
            ResBlock(feats, feats, 1),
            ResBlock(feats, feats, 2),
            ResBlock(feats, feats, 4),
            ResBlock(feats, feats, 2)
        ])

        self.embed = PositionalEncoding(feats)

        self.proj = nn.ModuleList([
            Projection(feats, feats),
            Projection(feats, feats),
            Projection(feats, feats),
            Projection(feats, feats)
        ])

        self.conv_out = Conv1d(feats, 2, 9, padding=4, padding_mode='reflect')
        self.dwt = DWT_1D(w_name)
        self.idwt = IDWT_1D(w_name)

    def forward(self, x, cond, noise_scale):
        xl, xh = self.dwt(x)
        x = torch.cat([xl, xh], dim=1)
        xl, xh = self.dwt(cond)
        cond = torch.cat([xl, xh], dim=1)
        noise_embed = self.embed(noise_scale)
        xs = []
        for layer, br in zip(self.stream_x, self.proj):
            x = layer(x)
            xs.append(br(x, noise_embed))

        for x, layer in zip(xs, self.stream_cond):
            cond = layer(cond) + x
        cond = self.conv_out(cond)
        xl, xh = cond[:, 0, :].unsqueeze(1), cond[:, 1, :].unsqueeze(1)

        return self.idwt(xl, xh)

class Model(nn.Module):
    def __init__(self, feats=64):
        super(Model, self).__init__()
        self.stream_x = nn.ModuleList([
            nn.Sequential(Conv1d(1, feats, 9, padding=4, padding_mode='reflect'),
                          nn.LeakyReLU(0.2)),
            ResBlock(feats, feats, 1),
            ResBlock(feats, feats, 2),
            ResBlock(feats, feats, 4),
            ResBlock(feats, feats, 2)
        ])

        self.stream_cond = nn.ModuleList([
            nn.Sequential(Conv1d(1, feats, 9, padding=4, padding_mode='reflect'),
                          nn.LeakyReLU(0.2)),
            ResBlock(feats, feats, 1),
            ResBlock(feats, feats, 2),
            ResBlock(feats, feats, 4),
            ResBlock(feats, feats, 2)
        ])

        self.embed = PositionalEncoding(feats)

        self.proj = nn.ModuleList([
            Projection(feats, feats),
            Projection(feats, feats),
            Projection(feats, feats),
            Projection(feats, feats)
        ])

        self.conv_out = Conv1d(feats, 1, 9, padding=4, padding_mode='reflect')

    def forward(self, x, cond, noise_scale):
        noise_embed = self.embed(noise_scale)
        xs = []
        for layer, br in zip(self.stream_x, self.proj):
            x = layer(x)
            xs.append(br(x, noise_embed))

        for x, layer in zip(xs, self.stream_cond):
            cond = layer(cond) + x

        return self.conv_out(cond)
