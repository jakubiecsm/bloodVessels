from numpy.matlib import ones
from pylab import *
import skimage
from skimage import exposure
from skimage.filters import rank
from skimage.color import rgb2hsv
from skimage.filters.edges import convolve
import numpy as np
from skimage.morphology import disk
from skimage.filters import frangi
from sklearn.metrics import confusion_matrix, f1_score
from imblearn.metrics import classification_report_imbalanced

class SimpleImageProcessor:

    def __init__(self):
        self.base_image = None
        self.expert_image = None
        self.green_image = None
        self.gray_image = None
        self.normalized_image = None
        self.normalized_expert_image = None
        self.rescaled_image = None
        self.frangied_image = None
        self.binary_image = None

    def load_base_image(self, base_image):
        self.base_image = base_image

    def load_expert_image(self, expert_image):
        self.expert_image = expert_image

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

    def calculate_effectiveness(self):
        img = self.binary_image.astype(np.uint8)
        img = np.array(img).reshape(-1)
        expert_img = self.expert_image
        expert_img = np.divide(expert_img, 255).astype(np.uint8)
        expert_img = np.array(expert_img).reshape(-1)
        print("Classification Report Imbalanced")
        print(classification_report_imbalanced(img, expert_img))

        conf_matrix = confusion_matrix(expert_img, img)
        true_positive = conf_matrix[0][0]
        false_positive = conf_matrix[0][1]
        false_negative = conf_matrix[1][0]
        true_negative = conf_matrix[1][1]
        print("TruePositive: {}, FalsePositive: {}, FalseNegative: {}, TrueNegative:{}"
              .format(true_positive, false_positive, false_negative, true_negative))

        accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)
        sensitivity = true_positive / (true_positive + false_negative)
        specificity = true_negative / (true_negative + false_positive) if (true_negative + false_positive) != 0 else 0
        print("Accuracy: {}, Sensitivity: {}, Specificity: {}".format(accuracy, sensitivity, specificity))

        f1 = f1_score(img, expert_img)
        print("F1_score: {}".format(f1))

    def process_image(self):
        self.extract_green()
        self.convert_to_gray()
        self.normalize()
        self.rescale_intensity()
        self.frangi_filter()
        self.background_cut_off()
        self.calculate_effectiveness()
