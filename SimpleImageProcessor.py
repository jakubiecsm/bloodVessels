from pylab import *
import skimage
from skimage import exposure
from skimage.filters import rank
from skimage.color import rgb2hsv
from skimage.filters.edges import convolve
import numpy as np
from skimage.morphology import disk
from skimage.filters import frangi


class SimpleImageProcessor:

    def __init__(self):
        self.base_image = None
        self.green_image = None
        self.gray_image = None
        self.normalized_image = None
        self.rescaled_image = None
        self.frangied_image = None
        self.binary_image = None

    def load_base_image(self, base_image):
        self.base_image = base_image

    def extract_green(self):
        self.green_image = self.base_image.copy()
        self.green_image[:, :, 0] = 0
        self.green_image[:, :, 2] = 0

    def convert_to_gray(self):
        self.gray_image = skimage.color.rgb2gray(self.green_image.copy())

    def normalize(self):
        self.normalized_image = exposure.equalize_hist(self.gray_image.copy())
        self.normalized_image = rank.equalize(skimage.util.img_as_ubyte(self.normalized_image), selem=disk(9))

    def rescale_intensity(self):
        K = ones([14, 14])
        K = K / sum(K)
        image = convolve(self.normalized_image, K)
        p_low, p_high = np.percentile(image, (5, 95))
        image = exposure.rescale_intensity(image, in_range=(p_low, p_high))
        self.rescaled_image = convolve(image, K)

    def frangi_filter(self):
        K = ones([14, 14])
        K = K / sum(K)
        self.frangied_image = convolve(frangi(self.rescaled_image), K)

    def background_cut_off(self):
        self.binary_image = self.frangied_image > np.mean(self.frangied_image) + 1.5 * np.std(self.frangied_image)

    def process_image(self):
        self.extract_green()
        self.convert_to_gray()
        self.normalize()
        self.rescale_intensity()
        self.frangi_filter()
        self.background_cut_off()
