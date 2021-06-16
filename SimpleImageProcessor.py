from pylab import *
import skimage
from skimage import exposure
from skimage.filters import rank
from skimage.color import rgb2hsv
from skimage.filters.edges import convolve
import numpy as np
from skimage.morphology import disk
from skimage.filters import frangi


def extract_green(image):
    green_image = image.copy()
    green_image[:, :, 0] = 0
    green_image[:, :, 2] = 0
    return green_image


def convert_to_gray(image):
    return skimage.color.rgb2gray(image)


def normalize(image, size):
    image = exposure.equalize_hist(image)
    image = rank.equalize(image, selem=disk(size))
    return image


def background_cut_off(image, cut_off):
    return image > np.mean(image) + cut_off/10 * np.std(image)


def process_image(image, cut_off, selem_size=9, convolve_size=14):
    image = normalize(convert_to_gray(extract_green(image)), selem_size)
    K = ones([convolve_size, convolve_size])
    K = K / sum(K)
    image = convolve(image, K)
    p_low, p_high = np.percentile(image, (5, 95))
    image = exposure.rescale_intensity(image, in_range=(p_low, p_high))
    # return frangi(convolve(image, K))
    return background_cut_off(frangi(convolve(image, K)), cut_off)

