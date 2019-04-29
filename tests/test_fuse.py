import unittest
import numpy as np
import cnnbin as module
from scipy.signal.windows import hann


class TestFuse(unittest.TestCase):
    def test_divide(self):
        image = np.random.random((50, 51)).astype("float32")
        block_shape = (8, 9)

        subimages = module.patches.split(image, block_shape)
        image_combined = module.patches.combine(subimages, image.shape)
        np.testing.assert_array_almost_equal(image, image_combined)

    def test_divide_color(self):
        image = np.random.random((50, 51, 3)).astype("float32")
        block_shape = (8, 9)

        subimages = module.patches.split(image, block_shape)
        image_combined = module.patches.combine(subimages, image.shape)
        np.testing.assert_array_almost_equal(image, image_combined)

    def test_fail_number_of_blocks(self):
        image = np.random.random((50, 51)).astype("float32")
        block_shape = (8, 9)

        subimages = module.patches.split(image, block_shape)
        with self.assertRaises(AssertionError):
            module.patches.combine(subimages[:-1], image.shape)

    def test_fail_wrong_shape(self):
        image = np.random.random((50, 51)).astype("float32")
        block_shape = (8, 9)

        subimages = module.patches.split(image, block_shape)
        with self.assertRaises(AssertionError):
            module.patches.combine(subimages[:-1], (123, 12))

    def test_sampling(self):
        image = np.random.random((360, 250, 3)).astype("float32")
        block_shape = (128, 128)
        subimages = module.patches.split(image, block_shape, sampling=1.1)
        image_combined = module.patches.combine(subimages, image.shape, sampling=1.1)
        np.testing.assert_array_almost_equal(image, image_combined)

    def test_window(self):
        image = np.random.random((360, 250, 3)).astype("float32")
        block_shape = (128, 128)
        subimages = module.patches.split(image, block_shape)
        image_combined = module.patches.combine(subimages, image.shape, windowfunc=hann)
        np.testing.assert_array_almost_equal(image, image_combined)


if __name__ == "__main__":
    unittest.main()
