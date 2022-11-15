import numpy as np
import torch
import math

from b_spline import B_spline
from scipy.stats import multivariate_normal
# Define element-wise erf for numpy arrays
erf = np.vectorize(math.erf)

def bake_cake(orientation, angle, resolution=(256,256), freq_min=.1, freq_max=.7, cake_type='basic', mean=[.0, .5], std=[[.005,0.0],[0.0,.0025]]):
    '''
        Generate cake wavelet mask
    '''
    
    X,Y = create_mesh(resolution)

    radial_coord, angular_coord = transform_cartesian_to_radial(X,Y)
    
    assert radial_coord.shape == angular_coord.shape
    assert np.alltrue(angular_coord>0)

    if cake_type == 'basic':
        values = basic_cake(radial_coord, angular_coord, orientation, angle, freq_max)
    elif cake_type == 'b_spline':
        values = b_spline_cake(radial_coord, angular_coord, orientation, angle, freq_max)
    elif cake_type == 'basic_bp':
        values = basic_bandpass(radial_coord, angular_coord, orientation, angle, freq_min, freq_max)
    elif cake_type == 'gabor':
        values = gabor(X, Y, orientation, np.array(mean), np.array(std))

    X,Y = transform_radial_to_cartesian(radial_coord, angular_coord)
    return np.stack([X, Y, values])

def create_mesh(resolution):
    '''
        Create a coordinate grid
    '''
    x_dim, y_dim = resolution
    
    X = np.linspace(-1, 1, x_dim)
    X = X[:, None].repeat(y_dim, axis=1)

    Y = np.linspace(-1, 1, y_dim)
    Y = Y[None, :].repeat(X.shape[0], axis=0)
    return X, Y

def b_spline_cake(  radial_coord, angular_coord, orientation, angle, freq_max, 
                    b_n = B_spline(4), radial_boundary_strength=2):
    '''
        Creates soft-boundary cake wavelet filters
    '''
    # angular resolution
    # equivalent to 2pi/N_slices
    s = angle

    # Shift center of spline to orientation
    shifted = (angular_coord-orientation)%(2*np.pi) - np.pi

    # Use erfc to create a soft radial boundary
    radial_boundary = (1-erf(radial_boundary_strength*(radial_coord-freq_max)))
    values = b_n(shifted/s) * radial_boundary
    
    return values

def basic_bandpass(radial_coord, angular_coord, orientation, angle, freq_min, freq_max):
    '''
        Create cake wavelets with hard boundaries
    '''
    values = np.zeros(angular_coord.shape)
    start = (orientation - angle/2) % (2*np.pi)
    end = (orientation + angle/2) % (2*np.pi)
    radial_idx = np.logical_and(freq_min<radial_coord, radial_coord<freq_max)
    
    if start>=end:
        # Catch if the angle overflows and loops around
        angle_idx = np.logical_or(start < angular_coord, angular_coord < end)
        
        idx = np.logical_and(angle_idx, radial_idx)
        values[idx] = 1
    else:
        # Between start and last
        angle_idx = np.logical_and(start < angular_coord, angular_coord < end)
        idx = np.logical_and(angle_idx,radial_idx)
        values[idx] = 1
    return values
def rotation_matrix(angle):
    '''
        Input:
            Angle - Rotation angle in Radians
        Output:
            mat - 2x2 Rotation matrix
    '''
    c = np.cos(angle)
    s = np.sin(angle)
    mat = np.array([[c , s],
                     [-s, c]])
    return mat

def gabor(X, Y, orientation, mean, cov):
    '''
        Create cake wavelets with hard boundaries
    '''
    values = np.zeros(X.shape)
    rot = rotation_matrix(orientation)
    mean = rot@mean
    # ?
    cov = rot @ cov @ rot.T
    coord = np.stack([X,Y], axis=-1)
    values = multivariate_normal(mean, cov).pdf(coord)
    return values

def basic_cake(radial_coord, angular_coord, orientation, angle, freq_max):
    '''
        Create cake wavelets with hard boundaries
    '''
    values = np.zeros(angular_coord.shape)
    start = (orientation - angle/2) % (2*np.pi)
    end = (orientation + angle/2) % (2*np.pi)
    if start>=end:
        # Catch if the angle overflows and loops around
        angle_idx = np.logical_or(start < angular_coord, angular_coord < end)
        radial_idx = radial_coord<freq_max
        idx = np.logical_and(angle_idx, radial_idx)
        values[idx] = 1
    else:
        # Between start and last
        angle_idx = np.logical_and(start < angular_coord, angular_coord < end)
        radial_idx = radial_coord<freq_max
        idx = np.logical_and(angle_idx,radial_idx)
        values[idx] = 1
    return values

def calculate_boundaries(max_radius, slices, radius_ticks):
    '''
        Calculates the orientation points for visualization
    '''
    radial_coord = np.arange(0, max_radius, radius_ticks)
    radial_coord = radial_coord[:, None].repeat(slices, axis=1)

    angular_coord = np.linspace(0, 360, slices, endpoint=False)
    angular_coord = angular_coord[:, None].repeat(radial_coord.shape[0], axis=1).T

    assert radial_coord.shape == angular_coord.shape

    values = np.ones(angular_coord.shape)

    X,Y = transform_radial_to_cartesian(radial_coord, angular_coord, degrees=True)
    
    return np.stack([X, Y])

def transform_cartesian_to_radial(X, Y):
    '''
        Convert to polar coordinates
        Input:
            X : x-domain discretized into N bins repeated M times (N x M)
            Y : y-domain discretized into M bins repeated N times (N x M)
        Return:
            r : The corresponding radial coordinate
            theta : The corresponding angular coordinate
    '''
    r = np.sqrt(X**2 + Y**2)
    ang = np.arctan2(X,Y) + np.pi
    return r, ang

def transform_radial_to_cartesian(rad, ang, degrees=False):
    '''
        Convert to Cartesian coordinates
        Input:
            r : The corresponding radial coordinate
            theta : The corresponding angular coordinate
        Return:
            X : x coordinate
            Y : y coordinate
    '''
    ang = ang
    if degrees==True:
        ang = np.deg2rad(ang)
    X = rad*np.cos(ang)
    Y = rad*np.sin(ang)
    return X, Y

def create_filter_bank(N_slices : int, resolution : tuple, wavelet_type : str):
    '''
        Input:
            N_slices : 2pi/angle. Denoted as S.
            resolution : Shape of image (N, M)
        Output:
            filters : 1 x 1 x S x N x M Tensor for elementwise product with batch
    '''
    angle = 2*np.pi/N_slices
    angles = np.linspace(0, 2*np.pi, N_slices, endpoint=False)
    filters = []
    # Create a filter for every orientation of a slice
    # This is equivalent to rotating the image and applying it to the image
    for orientation in angles:
        X, Y, val = bake_cake(orientation, angle, cake_type=wavelet_type, resolution=resolution)
        # Transform to match fft convensions
        val = transform_for_ifft(val)
        filters += [torch.tensor(val)]
    filters = torch.stack(filters).unsqueeze(0).unsqueeze(0)
    return filters

def transform_for_ifft(img):
    '''
        ifft expects positive frequencies 0:N followed by negative frequencies 0:N
    '''
    shape = img.shape
    N, M = shape
    img_2 = np.zeros(shape)
    pp_freqs = img[N//2:, M//2:]
    pn_freqs = img[N//2:, :M//2]
    np_freqs = img[:N//2, M//2:]
    nn_freqs = img[:N//2, :M//2]
    
    img_2 = np.zeros(shape)
    img_2[:N//2, :M//2] = pp_freqs
    img_2[:N//2, M//2:] = pn_freqs
    img_2[N//2:, :M//2] = np_freqs
    img_2[N//2:, M//2:] = nn_freqs
    return img_2