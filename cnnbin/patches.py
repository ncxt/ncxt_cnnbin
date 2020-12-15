"""
Handling of dividing and combining image pathces
"""
import functools
import operator
from math import ceil

import numpy as np

from .utils import window


def divide(length, block_size, n_blocks, ind_div=1):
    """ Divide a 1d array into evenly spaced
        blocks of size block_size.

    Arguments:
        length {[type]} -- Length of the space
        block_size {[type]} -- Size of the blocks
        n_blocks {[type]} -- Number of blocks
        ind_div {[int]} -- Make indecies divisable by argument

    Returns:
        [type] -- Starting index of the blocks
    """

    x0, xend = 0, length - block_size
    return np.round(np.linspace(x0, xend, n_blocks) / ind_div).astype(int) * ind_div


def num_blocks(lenght, size, sampling=1.0):
    """Number of blocks needed to cover the lenght

    Arguments:
        lenght {int} -- Length of the array
        size {int} -- blocks size
        sampling {float} -- sampling of the length
            1.0 gives the minimum required blocks to cover the array (default: {1.0})
        ind_div {[int]} -- Make indecies divisable by argument
    """
    return int(ceil(sampling * lenght / size))


def split(image, block_size, sampling=1.0, ind_div=1):
    """Split the volume into subvolumes of size block_shape

    Arguments:
        image {ndarray} -- 2d image
        block_size {tuple} -- shape of the blocks
        sampling {float} -- sampling of the length
            1.0 gives the minimum required blocks to cover the array (default: {1.0})
        ind_div {[int]} -- Make indecies divisable by argument

    Keyword Arguments:
        sampling {float} -- (over)Sampling of the data.
        1.0 produces the minimum number of blocks to cover the whole volume(default: {1.0})
    Returns:
        [list] -- List of subvolume Slice views of the original data
    """
    assert len(block_size) == 2, "block_size must have a length of 2"

    limits = [
        divide(length, size, num_blocks(length, size, sampling), ind_div=ind_div)
        for length, size in zip(image.shape, block_size)
    ]

    image_blocks = []
    for lim0 in limits[0]:
        slice0 = slice(lim0, lim0 + block_size[0])
        for lim1 in limits[1]:
            slice1 = slice(lim1, lim1 + block_size[1])
            image_blocks.append(image[slice0, slice1])

    return np.array(image_blocks)


def combine(blocklist, shape, sampling=1.0, windowfunc=None):
    """Combine patches into one image

    Arguments:
        blocklist {list} -- list of subimges
        shape {tuple} -- Original shape of image

    Keyword Arguments:
        sampling {float} -- Sampling used to produce blocks (default: {1.0})
        windowfunc {function} -- If defined, applies a windowing on the data for smoother
                                 blending (default: {None})

    Returns:
        [ndarray] -- Fused image
    """

    block_shape = blocklist[0].shape[:2]
    assert len(block_shape) == 2, "block_size must have a length of 2"

    required_blocks = [
        num_blocks(length, size, sampling) for length, size in zip(shape, block_shape)
    ]
    limits = [
        divide(length, size, num_blocks(length, size, sampling))
        for length, size in zip(shape, block_shape)
    ]

    assert len(blocklist) == functools.reduce(
        operator.mul, required_blocks, 1
    ), f"Number of blocks {len(blocklist)} does not match the argumetns {required_blocks}"

    image = np.zeros(shape, dtype="float32")
    image_n = np.zeros(shape[:2], dtype="float32")

    if windowfunc is None:
        index = 0
        for lim0 in limits[0]:
            slice0 = slice(lim0, lim0 + block_shape[0])
            for lim1 in limits[1]:
                slice1 = slice(lim1, lim1 + block_shape[1])

                image[slice0, slice1] += blocklist[index]
                image_n[slice0, slice1] += 1
                index += 1
    else:
        window_block = 0.01 + window(block_shape, windowfunc)
        index = 0
        for lim0 in limits[0]:
            slice0 = slice(lim0, lim0 + block_shape[0])
            for lim1 in limits[1]:
                slice1 = slice(lim1, lim1 + block_shape[1])

                if image.ndim == 3:
                    image[slice0, slice1] += (
                        blocklist[index] * window_block[:, :, np.newaxis]
                    )
                else:
                    image[slice0, slice1] += blocklist[index] * window_block

                image_n[slice0, slice1] += window_block
                index += 1

    if image.ndim == 3:
        return image / image_n[:, :, np.newaxis]
    return image / image_n
