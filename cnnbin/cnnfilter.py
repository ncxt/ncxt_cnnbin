"""
Noise 2 Noise binning of images
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.signal.windows import hann
from tqdm import tqdm

from .dataset import N2NMultiPatches, N2NPatches
from .patches import combine, split
from .torchsummary import bytes_2_human_readable
from .unetn2n import UNet
from .utils import progress_bar, rangebar, psnr, split_diagonal, split_diagonal_rgb


def _model_psnr(im1, im2, out_cnn1, out_cnn2, reference_image=None):
    im1, im2 = im1.cpu().detach().numpy(), im2.cpu().detach().numpy()

    if reference_image is not None:
        psnr_range = np.max(reference_image) - np.min(reference_image)
    else:
        maxval = max([np.max(im1), np.max(im2)])
        minval = max([np.min(im1), np.min(im2)])
        psnr_range = maxval - minval

    psnr_cnn = psnr(out_cnn1, out_cnn2, psnr_range)
    psnr_input = psnr(im1, im2, psnr_range)

    return psnr_cnn, psnr_input


class CNNbin:
    """
    Container of the Noise2Noise UNET for downsampling of images
    """

    def __init__(
        self,
        multichannel=False,
        depth=4,
        start_filts=48,
        block_size=(256, 256),
        batch_size=4,
        input_skip=True,
    ):

        self.multichannel = multichannel
        self.channels = 1
        self.block_size = block_size
        self.batch_size = batch_size

        if self.multichannel:
            self.channels = 3

        self.model = UNet(
            in_channels=self.channels,
            depth=depth,
            start_filts=start_filts,
            input_skip=input_skip,
        )

        self.criterion = torch.nn.MSELoss()
        self.model.cuda()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

        self.res_psnr = None
        self.res_loss = None

        self.epoch = 0

    def summary(self):
        """Print a summary of the CNN model

        Returns:
            float -- estimated size in bytes
        """

        self.model.cpu()
        size = self.batch_size * self.model.summary(
            (self.channels, self.block_size[0] // 2, self.block_size[1] // 2)
        )
        self.model.cuda()
        print(f"Total approximated size of : {bytes_2_human_readable(size)}")

    def _batch_to_torch(self, batch):
        if self.multichannel:
            batch = np.transpose(batch, (0, 3, 1, 2))
        else:
            if batch.ndim == 2:
                batch = batch.reshape(1, *batch.shape)
            batch = np.transpose(batch.reshape(1, *batch.shape), (1, 0, 2, 3))

        return torch.from_numpy(batch.astype("float32")).cuda()

    def _image_to_torch(self, image):
        if self.multichannel:
            image = np.transpose(image, (2, 0, 1))
        else:
            image = image.reshape(1, *image.shape)

        return torch.from_numpy(image.astype("float32")).view(1, *image.shape).cuda()

    def _sgd_step(self, torch_im1, torch_im2, alpha=0.95):
        # ===================forward=====================
        output1 = self.model(torch_im1)
        estimate = torch_im1 * (1 - alpha) + torch_im2 * alpha
        loss = self.criterion(output1, estimate)
        # ===================backward====================
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        out1 = output1.cpu().detach().numpy()

        if self.multichannel:
            return np.transpose(out1, (0, 3, 1, 2)), loss.item()
        else:
            return out1, loss.item()

    def _assert_image(self, image, even=True):
        if even:
            assert all(
                d % 2 == 0 for d in image.shape[:2]
            ), "Image shape must be divisible by 2"

        assert all(
            l >= d for l, d in zip(image.shape, self.block_size)
        ), "Image shape {} must be larger than block_size {}".format(
            image.shape, self.block_size
        )

        if self.multichannel:
            assert image.shape[2] == 3, "Colour images should be in format (H,W,C)"

    def train_image(self, image, num_epochs=10, learning_rate=1e-3, alpha=0.95):
        """Train single image

        Args:
            image (ndarray): Input image
            num_epochs (int, optional): Defaults to 10. Number of epochs
            learning_rate (float, optional): Defaults to 1e-3. Learning rate of optimizer
            alpha (float, optional): Defaults to 0.95. Regularization of filter to compensate for
            the non-exact loss function(0.5 converges to identity function, 1.0 is full smoothing)
        """

        self._assert_image(image)

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = learning_rate

        self.res_loss = []
        self.res_psnr = []

        self.model.train()
        pbar = tqdm(range(num_epochs), ncols=60, desc="PSNR {}/{}".format(0, 0))

        if self.multichannel:
            im1, im2 = split_diagonal_rgb(image)
        else:
            im1, im2 = split_diagonal(image)

        im1 = self._image_to_torch(im1)
        im2 = self._image_to_torch(im2)

        for _ in pbar:
            self.epoch += 1

            out_cnn1, loss1 = self._sgd_step(im1, im2, alpha)
            out_cnn2, loss2 = self._sgd_step(im2, im1, alpha)

            psnr_cnn, psnr_input = _model_psnr(im1, im2, out_cnn1, out_cnn2, image)

            pbar.set_description("PSNR ({:.3}/{:.3})".format(psnr_cnn, psnr_input))

            self.res_loss.append((loss1 + loss2))
            self.res_psnr.append(psnr_cnn)

    def train_random(
        self, image, samples=1, num_epochs=10, learning_rate=1e-3, alpha=0.95
    ):
        """Train single image on randomly sampled patches

        Args:
            image (ndarray): Input image
            num_batches (int, optional): Defaults to 4. Total number of minibatches
            num_epochs (int, optional): Defaults to 10. Number of epochs
            learning_rate (float, optional): Defaults to 1e-3. Learning rate of optimizer
            alpha (float, optional): Defaults to 0.95. Regularization of filter to compensate for
            the non-exact loss function(0.5 converges to identity function, 1.0 is full smoothing)
        """

        self._assert_image(image, even=False)

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = learning_rate

        self.res_loss = []
        self.res_psnr = []

        self.model.train()

        pbar = progress_bar(num_epochs, "PSNR {}/{}".format(0, 0))

        for _ in pbar:
            self.epoch += 1
            psnr_ref = []
            psnr_res = []
            losses = []

            dataset = N2NPatches(
                image,
                block_shape=self.block_size,
                sampling=samples,
                random=True,
                random_seed=self.epoch,
            )

            data_loader = torch.utils.data.DataLoader(
                dataset, batch_size=self.batch_size
            )

            for sample_batched in data_loader:
                im1, im2 = sample_batched

                out_cnn1, loss1 = self._sgd_step(im1, im2, alpha)
                out_cnn2, loss2 = self._sgd_step(im2, im1, alpha)

                psnr_cnn, psnr_input = _model_psnr(im1, im2, out_cnn1, out_cnn2, image)
                losses.append(loss1)
                losses.append(loss2)
                psnr_res.append(psnr_cnn)
                psnr_ref.append(psnr_input)

            pbar.set_description(
                "PSNR ({:.3}/{:.3})".format(np.mean(psnr_res), np.mean(psnr_ref))
            )

            self.res_loss.append(np.mean(losses))
            self.res_psnr.append(np.mean(psnr_res))

    def train_split(
        self, image, sampling=1.1, num_epochs=10, learning_rate=1e-3, alpha=0.95
    ):
        """Train single image on deterministic patches

        Args:
            image (ndarray): Input image
            sampling (int, optional): Defaults to 1.1. Oversampling of image for overlapping pathces
            num_epochs (int, optional): Defaults to 10. Number of epochs
            learning_rate (float, optional): Defaults to 1e-3. Learning rate of optimizer
            alpha (float, optional): Defaults to 0.95. Regularization of filter to compensate for
            the non-exact loss function(0.5 converges to identity function, 1.0 is full smoothing)
        """

        self._assert_image(image)

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = learning_rate

        self.res_loss = []
        self.res_psnr = []

        self.model.train()

        dataset = N2NPatches(image, block_shape=self.block_size, sampling=sampling)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size)

        pbar = progress_bar(num_epochs, "PSNR {}/{}".format(0, 0))
        for _ in pbar:
            self.epoch += 1
            psnr_ref = []
            psnr_res = []
            losses = []

            for sample_batched in data_loader:
                im1, im2 = sample_batched

                out_cnn1, loss1 = self._sgd_step(im1, im2, alpha)
                out_cnn2, loss2 = self._sgd_step(im2, im1, alpha)
                psnr_cnn, psnr_input = _model_psnr(im1, im2, out_cnn1, out_cnn2, image)
                losses.append(loss1)
                losses.append(loss2)
                psnr_res.append(psnr_cnn)
                psnr_ref.append(psnr_input)

            pbar.set_description(
                "PSNR ({:.3}/{:.3})".format(np.mean(psnr_res), np.mean(psnr_ref))
            )

            self.res_loss.append(np.mean(losses))
            self.res_psnr.append(np.mean(psnr_res))

    def train_list(self, images, samples=1, num_epochs=10, lr=1e-3, alpha=0.95):
        """ train using a list of images """

        for image in images:
            self._assert_image(image, even=False)

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        self.res_loss = []
        self.res_psnr = []

        self.model.train()
        pbar = progress_bar(num_epochs, desc="PSNR ({:.3}/{:.3})".format(0.0, 0.0))

        for _ in pbar:
            self.epoch += 1
            psnr_ref = []
            psnr_res = []
            losses = []

            dataset = N2NMultiPatches(
                images,
                block_shape=self.block_size,
                sampling=samples,
                random_seed=self.epoch,
            )

            data_loader = torch.utils.data.DataLoader(
                dataset, batch_size=self.batch_size
            )

            for _, sample_batched in enumerate(data_loader):
                im1, im2 = sample_batched

                out_cnn1, loss1 = self._sgd_step(im1, im2, alpha)
                out_cnn2, loss2 = self._sgd_step(im2, im1, alpha)

                psnr_cnn, psnr_input = _model_psnr(im1, im2, out_cnn1, out_cnn2)
                losses.append(loss1)
                losses.append(loss2)
                psnr_res.append(psnr_cnn)
                psnr_ref.append(psnr_input)

            pbar.set_description(
                "PSNR ({:.3}/{:.3})".format(np.mean(psnr_res), np.mean(psnr_ref))
            )

            self.res_loss.append(np.mean(losses))
            self.res_psnr.append(np.mean(psnr_res))

    def filter_patch(self, image):
        """ filter image patch using the network"""
        self._assert_image(image, even=True)
        self.model.eval()

        if self.multichannel:
            patch1, patch2 = split_diagonal_rgb(image)
        else:
            patch1, patch2 = split_diagonal(image)

        with torch.no_grad():
            torch_im1 = self._image_to_torch(patch1)
            torch_im2 = self._image_to_torch(patch2)

            output1 = self.model(torch_im1).cpu().detach().numpy()
            output2 = self.model(torch_im2).cpu().detach().numpy()

        if self.multichannel:
            retval = (output1[0, :] + output2[0, :]) / 2
            retval = np.transpose(retval, (1, 2, 0))
        else:
            retval = (output1[0, 0, :] + output2[0, 0, :]) / 2

        return retval

    def filter(self, image, sampling=1.1):
        """ filter image using the network by splitting it up into smaller pathces """
        self._assert_image(image, even=True)

        bin_block_size = [d // 2 for d in self.block_size]
        bin_shape = [d // 2 for d in image.shape]

        if self.multichannel:
            bin_shape[2] = image.shape[2]

        patches = split(image, self.block_size, sampling=sampling, ind_div=2)

        if self.multichannel:
            patches_filt = np.zeros((patches.shape[0], *bin_block_size, 3))
        else:
            patches_filt = np.zeros((patches.shape[0], *bin_block_size))

        for index in rangebar(len(patches)):
            patches_filt[index] = self.filter_patch(patches[index])

        return combine(patches_filt, bin_shape, sampling=sampling, windowfunc=hann)

    def load(self, filename):
        """ load model weights """
        self.model.load_state_dict(torch.load(filename))
        print("Net loaded")

    def save(self, filename):
        """ save model weights """
        torch.save(self.model.state_dict(), filename)
        print(f"Weights {filename} saved!")

    def plot_train(self):
        """ plot training progress """
        plt.figure(figsize=(8, 3))
        plt.subplot(121)
        plt.plot(self.res_loss)
        plt.ylabel("L2 Loss")
        plt.subplot(122)
        plt.plot(self.res_psnr)
        plt.ylabel("PSNR")
        plt.show()
