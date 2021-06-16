from PIL import Image
import glob
import cv2
from skimage import io


def load_images(directory_path, extension):
    image_list = []
    for filename in glob.glob(directory_path + '/*.' + extension):
        im = io.imread(filename)
        image_list.append(im)

    return image_list
