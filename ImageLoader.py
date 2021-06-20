import glob
from skimage import io

class ImageLoader:
    def __init__(self):
        self.healthy_path = 'healthy/'
        self.healthy_gold_path = 'healthyGoldStandard/'
        self.base_images = []
        self.init_base_images()
        self.selected_image = None
        self.corresponding_image = None

    def init_base_images(self):
        for i in range(15):
            filename = str(i + 1) + "_h.jpg"
            self.base_images.append((i+1, filename))

    def load_image(self, filename):
        self.selected_image = io.imread(self.healthy_path + filename)
        corresponding_filename = str.replace(filename, "jpg", "tif")
        self.corresponding_image = io.imread(self.healthy_gold_path + corresponding_filename)


def load_images(directory_path, extension):
    image_list = []
    for filename in glob.glob(directory_path + '/*.' + extension):
        im = io.imread(filename)
        image_list.append(im)

    return image_list
