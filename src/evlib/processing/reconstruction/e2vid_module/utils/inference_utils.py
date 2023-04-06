from typing import Any
from math import ceil, floor
from collections import deque

from torch.nn import ReflectionPad2d
import numpy as np
import torch
import cv2
import scipy.stats as st
import torch.nn.functional as F


def gkern(kernlen: int=5, nsig: float=1.0) -> torch.Tensor:
    """Returns a 2D Gaussian kernel array."""
    """https://stackoverflow.com/a/29731818"""
    interval = (2 * nsig + 1.) / (kernlen)
    x = np.linspace(-nsig - interval / 2., nsig + interval / 2., kernlen + 1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw / kernel_raw.sum()
    return torch.from_numpy(kernel).float()


class EventPreprocessor:
    """
    Utility class to preprocess event tensors.
    Can perform operations such as hot pixel removing, event tensor normalization,
    or flipping the event tensor.
    """

    def __init__(self, options: Any) -> None:

        print('== Event preprocessing ==')
        self.no_normalize = options.no_normalize
        if self.no_normalize:
            print('!!Will not normalize event tensors!!')
        else:
            print('Will normalize event tensors.')

        self.flip = options.flip
        if self.flip:
            print('Will flip event tensors.')

    def __call__(self, events: torch.Tensor) -> torch.Tensor:
        # Flip tensor vertically and horizontally
        if self.flip:
            events = torch.flip(events, dims=[2, 3])

        # Normalize the event tensor (voxel grid) so that
        # the mean and stddev of the nonzero values in the tensor are equal to (0.0, 1.0)
        if not self.no_normalize:
            nonzero_ev = (events != 0)
            num_nonzeros = nonzero_ev.sum()
            if num_nonzeros > 0:
                # compute mean and stddev of the **nonzero** elements of the event tensor
                # we do not use PyTorch's default mean() and std() functions since it's faster
                # to compute it by hand than applying those funcs to a masked array
                mean = events.sum() / num_nonzeros
                stddev = torch.sqrt((events ** 2).sum() / num_nonzeros - mean ** 2)
                mask = nonzero_ev.float()
                events = mask * (events - mean) / stddev

        return events


class IntensityRescaler:
    """
    Utility class to rescale image intensities to the range [0, 1],
    using (robust) min/max normalization.
    Optionally, the min/max bounds can be smoothed over a sliding window to avoid jitter.
    """

    def __init__(self, options: Any) -> None:
        self.auto_hdr = options.auto_hdr
        self.intensity_bounds = deque()  # type: ignore
        self.auto_hdr_median_filter_size = options.auto_hdr_median_filter_size
        self.Imin = options.Imin
        self.Imax = options.Imax

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """
        param img: [1 x 1 x H x W] Tensor taking values in [0, 1]
        """
        if self.auto_hdr:
            Imin = torch.min(img).item()
            Imax = torch.max(img).item()

            # ensure that the range is at least 0.1
            Imin = np.clip(Imin, 0.0, 0.45)
            Imax = np.clip(Imax, 0.55, 1.0)

            # adjust image dynamic range (i.e. its contrast)
            if len(self.intensity_bounds) > self.auto_hdr_median_filter_size:
                self.intensity_bounds.popleft()

            self.intensity_bounds.append((Imin, Imax))
            self.Imin = np.median([rmin for rmin, rmax in self.intensity_bounds])
            self.Imax = np.median([rmax for rmin, rmax in self.intensity_bounds])

        img = 255.0 * (img - self.Imin) / (self.Imax - self.Imin)
        img.clamp_(0.0, 255.0)
        img = img.byte()  # convert to 8-bit tensor

        return img


class UnsharpMaskFilter:
    """
    Utility class to perform unsharp mask filtering on reconstructed images.
    """
    def __init__(self, options: Any, device: torch.device) -> None:
        self.unsharp_mask_amount = options.unsharp_mask_amount
        self.unsharp_mask_sigma = options.unsharp_mask_sigma
        self.gaussian_kernel_size = 5
        self.gaussian_kernel = gkern(self.gaussian_kernel_size,
                                     self.unsharp_mask_sigma).unsqueeze(0).unsqueeze(0).to(device)

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if self.unsharp_mask_amount > 0:
            blurred = F.conv2d(img, self.gaussian_kernel,
                                padding=self.gaussian_kernel_size // 2)
            img = (1 + self.unsharp_mask_amount) * img - self.unsharp_mask_amount * blurred
        return img


class ImageFilter:
    """
    Utility class to perform some basic filtering on reconstructed images.
    """

    def __init__(self, options: Any) -> None:
        self.bilateral_filter_sigma = options.bilateral_filter_sigma

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if self.bilateral_filter_sigma:
            # TODO check - this looks like returning numpy, not torch.
            filtered_img = np.zeros_like(img)
            filtered_img = cv2.bilateralFilter(
                img, 5, 25.0 * self.bilateral_filter_sigma, 25.0 * self.bilateral_filter_sigma)
            img = filtered_img  # type: ignore
        return img


def optimal_crop_size(max_size: int, max_subsample_factor: int) -> int:
    """ Find the optimal crop size for a given max_size and subsample_factor.
        The optimal crop size is the smallest integer which is greater or equal than max_size,
        while being divisible by 2^max_subsample_factor.
    """
    crop_size = int(pow(2, max_subsample_factor) * ceil(max_size / pow(2, max_subsample_factor)))
    return crop_size


class CropParameters:
    """ Helper class to compute and store useful parameters for pre-processing and post-processing
        of images in and out of E2VID.
        Pre-processing: finding the best image size for the network, and padding the input image with zeros
        Post-processing: Crop the output image back to the original image size
    """

    def __init__(self, width: int, height: int, num_encoders: int) -> None:
        self.height = height
        self.width = width
        self.num_encoders = num_encoders
        self.width_crop_size = optimal_crop_size(self.width, num_encoders)
        self.height_crop_size = optimal_crop_size(self.height, num_encoders)

        self.padding_top = ceil(0.5 * (self.height_crop_size - self.height))
        self.padding_bottom = floor(0.5 * (self.height_crop_size - self.height))
        self.padding_left = ceil(0.5 * (self.width_crop_size - self.width))
        self.padding_right = floor(0.5 * (self.width_crop_size - self.width))
        self.pad = ReflectionPad2d((self.padding_left, self.padding_right, self.padding_top, self.padding_bottom))

        self.cx = floor(self.width_crop_size / 2)
        self.cy = floor(self.height_crop_size / 2)

        self.ix0 = self.cx - floor(self.width / 2)
        self.ix1 = self.cx + ceil(self.width / 2)
        self.iy0 = self.cy - floor(self.height / 2)
        self.iy1 = self.cy + ceil(self.height / 2)
