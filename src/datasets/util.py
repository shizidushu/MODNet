import random

import numpy as np
from PIL import Image
from scipy.ndimage import grey_dilation, grey_erosion
import cv2


def gen_trimap(alpha):
    if np.amax(alpha) > 1:
        alpha = alpha / 255.0
    foreground = alpha > 0
    ### 以下连续几行修复了，当alpha为全0时候出错，即没有前景的是时候
    res = None
    res = Image.fromarray(foreground).getbbox()
    if res is None:
        left, upper, right, ylower = 0, 0, alpha.shape[1], alpha.shape[0]
    else:
        left, upper, right, ylower = res

    bbox_size = ((right - left) + (ylower - upper)) // 2
    factor = int(np.round(bbox_size / 256))
    if factor == 0:
        factor = 1
    d_size = factor * random.randint(10, 20)  # dilate kernel size
    e_size = factor * random.randint(10, 20)  # erode kernel size
    
    trimap = (alpha >= 0.9).astype('float32')
    not_bg = (alpha > 0).astype('float32')
    trimap[np.where(
        (grey_dilation(not_bg, size=(d_size, d_size)) - grey_erosion(trimap, size=(e_size, e_size))) != 0)] = 0.5
    return trimap

def gen_trimap_with_dilate(alpha):
    if np.amax(alpha) <= 1:
        alpha = alpha * 255.0
    kernel_size = random.randint(15, 30)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size,kernel_size))
    fg_and_unknown = np.array(np.not_equal(alpha, 0).astype(np.float32))
    fg = np.array(np.equal(alpha, 255).astype(np.float32))
    dilate =  cv2.dilate(fg_and_unknown, kernel, iterations=1)
    erode = cv2.erode(fg, kernel, iterations=1)
    trimap = erode *255 + (dilate-erode)*127.5
    trimap = trimap / 255.0
    return trimap


def gen_trimap_with_dilate_bbox(alpha):
    if np.amax(alpha) <= 1:
        alpha = alpha * 255.0
    foreground = alpha > 0
    ### 以下连续几行修复了，当alpha为全0时候出错，即没有前景的是时候
    res = None
    res = Image.fromarray(foreground).getbbox()
    if res is None:
        left, upper, right, ylower = 0, 0, alpha.shape[1], alpha.shape[0]
    else:
        left, upper, right, ylower = res
    bbox_size = ((right - left) + (ylower - upper)) // 2
    factor = int(np.round(bbox_size / 256))
    if factor == 0:
        factor = 1
    d_size = factor * random.randint(10, 20)  # dilate kernel size
    e_size = factor * random.randint(10, 20)  # erode kernel size

    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d_size,d_size))
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (e_size,e_size))
    fg_and_unknown = np.array(np.not_equal(alpha, 0).astype(np.float32))
    fg = np.array(np.equal(alpha, 255).astype(np.float32))
    dilate =  cv2.dilate(fg_and_unknown, kernel_dilate, iterations=1)
    erode = cv2.erode(fg, kernel_erode, iterations=1)
    trimap = erode *255 + (dilate-erode)*127.5
    trimap = trimap / 255.0
    return trimap