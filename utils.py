import string

import cv2
import numpy as np
from PIL import Image

char_list = string.ascii_letters + string.digits


def encode_to_labels(txt):
    # encoding each output word into digits
    dig_lst = []
    for index, char in enumerate(txt):
        try:
            dig_lst.append(char_list.index(char))
        except:
            print(char)

    return dig_lst


def find_dominant_color(image):
    # Resizing parameters
    width, height = 150, 150
    image = image.resize((width, height), resample=0)
    # Get colors from image object
    pixels = image.getcolors(width * height)
    # Sort them by count number(first element of tuple)
    sorted_pixels = sorted(pixels, key=lambda t: t[0])
    # Get the most frequent color
    dominant_color = sorted_pixels[-1][1]
    return dominant_color


def preprocess_img(img, imgSize):
    "put img into target img of size imgSize and normalize gray-values"

    # In case of black images with no text just use black image instead.
    if img is None:
        img = np.zeros([imgSize[1], imgSize[0]])
        print("Image None!")

    # create target image and copy sample image into it
    (wt, ht) = imgSize
    (h, w) = img.shape
    fx = w / wt
    fy = h / ht
    f = max(fx, fy)
    newSize = (max(min(wt, int(w / f)), 1),
               max(min(ht, int(h / f)), 1))  # scale according to f (result at least 1 and at most wt or ht)
    img = cv2.resize(img, newSize, interpolation=cv2.INTER_CUBIC)  # INTER_CUBIC interpolation best approximate the pixels image
    # see this https://stackoverflow.com/a/57503843/7338066
    most_freq_pixel = find_dominant_color(Image.fromarray(img))
    target = np.ones([ht, wt]) * most_freq_pixel
    target[0:newSize[1], 0:newSize[0]] = img

    img = target

    return img
