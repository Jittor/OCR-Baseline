from PIL import Image
import numpy as np
import cv2
import random
import math

pillow_interp_codes = {
    'nearest': Image.NEAREST,
    'bilinear': Image.BILINEAR,
    'bicubic': Image.BICUBIC,
    'box': Image.BOX,
    'lanczos': Image.LANCZOS,
    'hamming': Image.HAMMING
}


def imresize(img, size, return_scale=False, interpolation='bilinear'):
    h, w = img.shape[:2]
    assert img.dtype == np.uint8
    pil_image = Image.fromarray(img)
    pil_image = pil_image.resize(size, pillow_interp_codes[interpolation])
    resized_img = np.array(pil_image)
    if not return_scale:
        return resized_img
    else:
        w_scale = size[0] / w
        h_scale = size[1] / h
        return resized_img, w_scale, h_scale


def convert_color_factory(src, dst):

    code = getattr(cv2, f'COLOR_{src.upper()}2{dst.upper()}')

    def convert_color(img):
        out_img = cv2.cvtColor(img, code)
        return out_img

    convert_color.__doc__ = f"""Convert a {src.upper()} image to {dst.upper()}
        image.
    Args:
        img (ndarray or str): The input image.
    Returns:
        ndarray: The converted {dst.upper()} image.
    """

    return convert_color


bgr2rgb = convert_color_factory('bgr', 'rgb')

rgb2bgr = convert_color_factory('rgb', 'bgr')

bgr2hsv = convert_color_factory('bgr', 'hsv')

hsv2bgr = convert_color_factory('hsv', 'bgr')

bgr2hls = convert_color_factory('bgr', 'hls')

hls2bgr = convert_color_factory('hls', 'bgr')


def crop(img, scale):
    img = Image.fromarray(np.uint8(img))
    w, h = img.size
    area = w * h
    ratio = 0.5 + random.random() * 1.5
    new_area = area * scale
    for _ in range(10):
        new_h = int(round(math.sqrt(new_area / ratio)))
        new_w = int(round(math.sqrt(new_area * ratio)))
        if new_h < h and new_w < w:
            new_h_start = random.randint(0, h - new_h)
            new_w_start = random.randint(0, w - new_w)
            break
    else:
        new_w = min(h, w)
        new_h = new_w
        new_h_start = (h - new_h) // 2
        new_w_start = (w - new_w) // 2
    start = [new_w_start, new_h_start]
    end = [new_w_start + new_w, new_h_start + new_h]
    img = img.crop((new_w_start, new_h_start, new_w_start +
                   new_w, new_h_start + new_h))
    return np.array(img), start + end
