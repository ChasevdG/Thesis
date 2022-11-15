import torch
from wavelets import create_filter_bank

class WaveletLiftingLayer(torch.nn.Module):
    def __init__(self, slices, resolution, wavelet_type='b_spline') -> None:
        super().__init__()
        # 1 x 1 x S x N x M
        self.register_buffer('filters', create_filter_bank(slices, resolution, wavelet_type))
        self._wavelet_type = wavelet_type
        
    def forward(self, x):
        '''
            Input:
                x - B x C x N x M Tensor
            Return
                y - B x C x S x N x M Tensor of 
        '''
        
        x_ = torch.fft.fft2(x)        
        x_ = x_.unsqueeze(-3) # B x C x 1 x N x M
        x_ = x_ * self.filters # B x C x S x N x M
        y = torch.fft.ifft2(x_)
        
        return torch.real(y).float()