import cv2
import numpy as np
from glob import glob


def process(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (240, 120))
    image = image[40:100,:]
    image = normalize(image)
    image = image[:, :, np.newaxis]
    return image


def normalize(image):
    '''
        Subtracting the mean from each pixel and then dividing the result by the standard deviatio.
    '''
    mean, stdv = [120.9934, 18.8303]
    return (image - mean) / stdv


def get_statistics_from(dataset_path):
    files = glob('{0}/*.png'.format(dataset_path))
    N = len(files)
    mean_img = None
    for index, img_path in enumerate(files):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = img[40:100,:]
        if mean_img is None:
            mean_img = img
        else:
            mean_img = mean_img + (img - mean_img) / (index + 2)
    # cv2.imshow('test', mean_img)
    # cv2.waitKey(0)
    m, n = mean_img.shape
    mean = np.sum(mean_img) / (m * n * 1.0)
    stdv = 0
    for index, img_path in enumerate(files):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = img[40:100,:]
        stdv += np.sum(np.abs(img - mean)) / (m * n)
    stdv = stdv / N
    print(mean, stdv)
    return mean, stdv


if __name__ == '__main__':
    get_statistics_from('_debug')
    # get_statistics_from('_out')
