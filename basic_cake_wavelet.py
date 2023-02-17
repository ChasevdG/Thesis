import numpy as np
import torch
from functools import partial

grid_sample = partial(
    torch.nn.functional.grid_sample,
    padding_mode='zeros',
    align_corners=True,
    mode='bilinear',
)

def rotate(grid, theta):
    rot_mat = torch.tensor([ [np.cos(theta), -np.sin(theta)],
               [np.sin(theta), np.cos(theta)]])
    return torch.matmul(rot_mat, grid.flatten(start_dim=1).double()).reshape(grid.shape)

def polar(grid):
    # Convert grid to polar coordinates
    x = grid[0]
    y = grid[1]

    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y,x)
    return np.array([r, theta])

def create_grid(length, rotation=0):
    #create mesh
    X = torch.tensor(np.linspace(-1,1,length, endpoint=True))
    Y = torch.tensor(np.linspace(-1,1,length, endpoint=True))
    X, Y = torch.meshgrid(X, Y)
    
    # Rotate
    grid = torch.concat([X.unsqueeze(0),Y.unsqueeze(0)], dim=0)
    grid = rotate(grid, rotation)

    _, N, M = grid.shape
    return grid.permute((1,2,0)).reshape((1,N,M,2))

def Lift_Kernel(kernel, n_rot):
    '''
    Lifts a kernel
    Input:
        Kernel is a 4D tensor of shape (B,C,N,M)
        n_rot is the number of rotations
    '''
    
    rotations = np.linspace(0,2*np.pi, n_rot, endpoint=False)
    N, M = kernel.shape[-2:]
    kernels = None
    for i, rot in enumerate(rotations):
        grid_i = create_grid(N, rot)
        f_1 = grid_sample(kernel, grid_i).unsqueeze(0)
        if kernels is None:
            kernels = f_1
        else:
            kernels = torch.cat((kernels, f_1), dim=0)
    return kernels.permute(1,2,0,3,4)

def create_cake(n_rot, resolution):
    grid = create_grid(resolution)
    X = grid[0,:,:,0]
    Y = grid[0,:,:,1]
    r, theta = polar([X,Y])

    # Check within angle and Nyquist frequency
    wave = ((r<=1) * (theta>=0) * (theta<np.pi/2)).double()
    # Fix DC
    wave[r==0] = 1/n_rot
    # Add batch and channel dimensions
    wave = wave.unsqueeze(0).unsqueeze(0)
    ls = Lift_Kernel(wave, 4)
    return ls
