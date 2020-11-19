import numpy as np
#import matplotlib.pyplot as plt
import math
import numbers
import torch
from torch import nn
from torch.nn import functional as F





class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, sigma, channels=1, kernel_size=10,  dim=2, normalize_output=False):
        super(GaussianSmoothing, self).__init__()
        #self.normalize = normalize_output
        if kernel_size % 2 == 0:
            int_size = (kernel_size // 2)
            self.pad_tuple = (int_size, int_size-1, int_size, int_size-1)
        else :
            int_size = (kernel_size // 2)
            self.pad_tuple = (int_size, int_size, int_size, int_size)

        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim

        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / std) ** 2 / 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input, useless_param=0):
        input = F.pad(input, self.pad_tuple, mode='constant')
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        input_filtered = self.conv(input, weight=self.weight, groups=self.groups)
        input_filtered = normalize_data(input_filtered)
        #if self.normalize :
        #    data_fl = torch.flatten(input_filtered, start_dim=1)
        #    out, _ = data_fl.max(dim=1)
        #    out = out.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        #    input_filtered /= out

        return input_filtered


def normalize_data(data, start_dim=1):

    data_fl = torch.flatten(data, start_dim=start_dim)
    out_min, _ = data_fl.min(dim=1)
    data_fl = data_fl - out_min[:, None]
    out_max, _ = data_fl.max(dim=1)
    data_fl = data_fl/out_max[:, None]
    return data_fl.view_as(data)

#def normalize_data(data):
#    data_fl = torch.flatten(data, start_dim=1)
#    out, _ = data_fl.max(dim=1)
#    out = out.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
#    return data/out

class WhiteNoise(nn.Module):
    def __init__(self, param):
        super(WhiteNoise, self).__init__()
        self.param = param

    def forward(self, x):
        input_filtered = x + self.param * torch.randn_like(x)
        return input_filtered

class SaltPepper(nn.Module):
    def __init__(self, param):
        super(SaltPepper, self).__init__()
        self.param = param

    def forward(self, x):
        to_out = x.clone()
        mask = torch.bernoulli(torch.ones_like(x) * self.param).bool()
        to_out[mask] = torch.bernoulli(torch.ones_like(x[mask]) * 0.5)
        return to_out




class WhitenMNIST(object):
    def __init__(self, matrix_path):
        matrix = np.load(matrix_path)
        self.matrix = torch.from_numpy(matrix).float()

    def __call__(self, img):
        shape = img.shape
        img = img.flatten() @ self.matrix 
        return img.reshape(shape) 