"""
derived torch.utils.dataDataset classes for handling training data
"""
import numpy as np
import torch
from torch.utils.data import Dataset
import random

from .patches import split
from .utils import random_patches, split_stack, split_stack_rgb


def augment_imagepair(image1, image2, multichannel):
    transpose = (1, 0, 2) if multichannel else (1, 0)
    for dim in range(2):
        if random.choice([True, False]):
            image1 = np.flip(image1, axis=dim)
            image2 = np.flip(image2, axis=dim)
    if random.choice([True, False]):
        image1 = image1.transpose(transpose)
        image2 = image2.transpose(transpose)
    return image1, image2


class N2NPatches(Dataset):
    """Noise2Noise Patches"""

    def __init__(
        self,
        image,
        block_shape,
        random=False,
        sampling=1.05,
        random_seed=1,
        totorch=True,
        augment=False,
    ):
        self.image = image
        self.block_shape = block_shape
        self.random = random
        self.sampling = sampling
        self.totorch = totorch
        self.augment = augment
        self.multichannel = False
        if image.ndim == 3:
            self.multichannel = True

        self.patches = None
        if random:
            self.patches = random_patches(
                self.image,
                self.block_shape,
                max_patches=self.sampling,
                random_state=random_seed,
            )
        else:
            self.patches = np.array(
                split(image, self.block_shape, sampling=self.sampling)
            )

        if self.multichannel:
            self.stack1, self.stack2 = split_stack_rgb(self.patches)
        else:
            self.stack1, self.stack2 = split_stack(self.patches)

    def image_to_torch(self, image):
        """convert numpy array to torch tensor"""
        if self.multichannel:
            image = np.transpose(image, (2, 0, 1))
        else:
            image = image.reshape(1, *image.shape)

        return torch.from_numpy(image.astype("float32")).cuda()

    def __getitem__(self, index):
        image1 = self.stack1[index]
        image2 = self.stack2[index]

        if self.augment:
            image1, image2 = augment_imagepair(image1, image2, self.multichannel)

        if self.totorch:
            torch_block1 = self.image_to_torch(image1)
            torch_block2 = self.image_to_torch(image2)
            return torch_block1, torch_block2

        return image1, image2

    def __len__(self):
        return len(self.patches)


class N2NMultiPatches(Dataset):
    """Noise2Noise Patches from a list of images"""

    def __init__(
        self,
        images,
        block_shape,
        sampling=1,
        random_seed=1,
        totorch=True,
        augment=False,
    ):
        self.images = images
        self.block_shape = block_shape
        self.sampling = sampling
        self.totorch = totorch
        self.augment = augment
        self.multichannel = False

        for image in images:
            assert image.ndim == images[0].ndim, "Images must be of the same mode"

        if images[0].ndim == 3:
            self.multichannel = True

        self.patches = np.concatenate(
            [
                random_patches(
                    im,
                    self.block_shape,
                    max_patches=self.sampling,
                    random_state=random_seed + i,
                )
                for i, im in enumerate(images)
            ],
            0,
        )

        if self.multichannel:
            self.stack1, self.stack2 = split_stack_rgb(self.patches)
        else:
            self.stack1, self.stack2 = split_stack(self.patches)

    def image_to_torch(self, image):
        """ convert numpy image to PyTorch tensor """
        if self.multichannel:
            image = np.transpose(image, (2, 0, 1))
        else:
            image = image.reshape(1, *image.shape)

        return torch.from_numpy(image.astype("float32")).cuda()

    def __getitem__(self, index):
        image1 = self.stack1[index]
        image2 = self.stack2[index]

        if self.augment:
            image1, image2 = augment_imagepair(image1, image2, self.multichannel)

        if self.totorch:
            torch_block1 = self.image_to_torch(image1)
            torch_block2 = self.image_to_torch(image2)
            return torch_block1, torch_block2

        return image1, image2

    def __len__(self):
        return len(self.patches)
