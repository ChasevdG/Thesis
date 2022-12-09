import torch
from wavelets import create_filter_bank
import matplotlib.pyplot as plt
import torchvision

def rotate_tensor(tensor, angle):
    return torchvision.transforms.functional.rotate(tensor, angle, torchvision.transforms.functional.InterpolationMode.BILINEAR)

class WaveletLiftingLayer(torch.nn.Module):
    def __init__(self, slices, resolution, wavelet_type='b_spline') -> None:
        super().__init__()
        # 1 x 1 x S x N x M
        self.register_buffer('filters', create_filter_bank(slices, resolution, wavelet_type))
        self._wavelet_type = wavelet_type
        
        for filter in self.filters.squeeze():
            plt.imshow(filter.detach().numpy())
            plt.show()
        for j,filter in enumerate(self.filters.squeeze()):
            for i,f in enumerate(self.filters.squeeze()):
                temp = torch.real(torch.fft.ifft2(f))
                temp = rotate_tensor(temp.unsqueeze(0), (i-j)*360/slices)
                plt.imshow((torch.real(torch.fft.ifft2(filter))-temp).squeeze())
                plt.show()
                assert torch.allclose(temp, torch.fft.ifft2(filter))
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
        
    
        return torch.abs(y).float()